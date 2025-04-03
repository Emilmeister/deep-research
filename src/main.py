import asyncio

from agents.models import openai_provider
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_default_openai_api, set_trace_processors, Runner, input_guardrail, \
    GuardrailFunctionOutput, InputGuardrailTripwireTriggered, trace
from openai.types.responses import EasyInputMessageParam

from phoenix.otel import register

from src.model.structured_outputs import TableOfConcepts, TableOfConceptsGuardrail, FollowUpQuestions, NewHypothesis, \
    ChapterText
from src.research_agents import TableOfConceptsAgent, TableOfConceptsGuardrailAgent, FollowUpQuestionsAgent, \
    HyposGeneratingAgent, ChapterEditorAgent, ChapterEditorSummaryAgent

import gradio

from src.tools import search_duckduckgo

# configure the Phoenix tracer
set_trace_processors([])
tracer_provider = register(
    project_name="deep-research",  # Default is 'default'
    endpoint='http://localhost:6006/v1/traces',
    auto_instrument=True
)

set_default_openai_client(AsyncOpenAI(base_url="http://94.41.23.12:11434/v1", api_key="1", timeout=60 * 5))
set_default_openai_api('chat_completions')
openai_provider.DEFAULT_MODEL = 'aya-expanse:32b'

table_of_concepts_guardrail_agent = TableOfConceptsGuardrailAgent()


@input_guardrail
async def table_of_concepts_guardrail(ctx, agent, input):
    if len(input) > 1:
        result = await Runner.run(table_of_concepts_guardrail_agent, input, context=ctx.context)
        output = TableOfConceptsGuardrail.model_validate(result.final_output)
        return GuardrailFunctionOutput(
            output_info=output.explanation,
            tripwire_triggered=output.satisfied,
        )
    else:
        return GuardrailFunctionOutput(
            output_info="len < 1",
            tripwire_triggered=False,
        )


table_of_concepts_agent = TableOfConceptsAgent()
follow_up_questions_agent = FollowUpQuestionsAgent()
hypos_agent = HyposGeneratingAgent()
chapter_editor_agent = ChapterEditorAgent()
chapter_editor_summary_agent = ChapterEditorSummaryAgent()
table_of_concepts_agent.input_guardrails.append(table_of_concepts_guardrail)

breadth_of_research = 1
depth_of_research = 1


async def main():
    with trace("Research workflow") as research_trace:
        research_trace.start()
        try:
            inp = input("User: ")
            inp_list = [EasyInputMessageParam(content=inp, role="user", type='message')]
            table_of_concepts = None
            while len(inp) > 0:
                result = await Runner.run(table_of_concepts_agent, inp_list)
                table_of_concepts = TableOfConcepts.model_validate(result.final_output)
                print(f"{table_of_concepts_agent.name}: ", table_of_concepts)
                inp = input("User: ")
                inp_list = result.to_input_list() + [EasyInputMessageParam(content=inp, role="user", type='message')]
        except InputGuardrailTripwireTriggered as e:
            # Оглавление готово
            done_chapters = {}
            for chapter in table_of_concepts.chapters:
                if chapter.need_research:
                    summaries = []
                    hypos = []
                    context = {
                        'title': table_of_concepts.title,
                        'chapter_name': chapter.chapter_name,
                        'chapter_description': chapter.chapter_description,
                        'summaries': summaries,
                        'hypos': hypos
                    }
                    for depth in range(depth_of_research):
                        result = await Runner.run(follow_up_questions_agent, [], context=context)
                        result = FollowUpQuestions.model_validate(result.final_output)

                        for i, question in enumerate(result.questions):
                            if i < breadth_of_research:
                                search = await search_duckduckgo(question)
                                summaries.append(search)

                        result = await Runner.run(hypos_agent, [], context=context)
                        result = NewHypothesis.model_validate(result.final_output)
                        hypos.extend(result.list_of_brilliant_ideas)
                    result = await Runner.run(chapter_editor_agent, [], context=context)
                    result = ChapterText.model_validate(result.final_output)
                    print("\n+++++\n", result)
                    done_chapters[chapter.chapter_name] = result.chapter_text_without_title_in_head

            for chapter in table_of_concepts.chapters:
                if not chapter.need_research:
                    context = {
                        'title': table_of_concepts.title,
                        'chapter_name': chapter.chapter_name,
                        'chapter_description': chapter.chapter_description,
                        'done_chapters': done_chapters
                    }
                    result = await Runner.run(chapter_editor_summary_agent, [], context=context)
                    result = ChapterText.model_validate(result.final_output)
                    print("\n------\n", result)
                    done_chapters[chapter.chapter_name] = result.chapter_text_without_title_in_head

            final_research = "\n\n".join(
                f"# {chapter.chapter_name}\n{done_chapters[chapter.chapter_name]}" for chapter in
                table_of_concepts.chapters)
            print("\n\n................\n\n", final_research)
        research_trace.finish()


if __name__ == "__main__":
    asyncio.run(main())
