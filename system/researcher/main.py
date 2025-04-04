from collections import defaultdict

from agents.models import openai_provider
from gradio import ChatMessage
from openai import AsyncOpenAI
from agents import set_default_openai_client, set_default_openai_api, set_trace_processors, Runner, input_guardrail, \
    GuardrailFunctionOutput, InputGuardrailTripwireTriggered, trace
from openai.types.responses import EasyInputMessageParam

from phoenix.otel import register

from system.researcher.model.structured_outputs import TableOfConcepts, TableOfConceptsGuardrail, FollowUpQuestions, NewHypothesis, \
    ChapterText
from system.researcher.research_agents import TableOfConceptsAgent, TableOfConceptsGuardrailAgent, FollowUpQuestionsAgent, \
    HyposGeneratingAgent, ChapterEditorAgent, ChapterEditorSummaryAgent
import gradio as gr

from system.researcher.tools import search_web, search_arxiv_relevant_pdfs_and_summarize

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


def to_openai_format(message, history):
    result = []
    for msg in history:
        result.append(EasyInputMessageParam(role=msg['role'], content=msg['content']))

    result.append(EasyInputMessageParam(role='user', content=message))

    return result


def to_gradio_format(history):
    result = []
    for message in history:
        result.append(ChatMessage(role=message['role'], content=message['content']))
    return result


def print_used_urls(urls):
    output = "## Написано на основании\n"
    if len(urls) == 0:
        return ""

    for i, url in enumerate(urls):
        output = output + f'{i+1}. {url}\n'

    return output


async def chat(message, history, table_of_concepts_json, breadth_of_research, depth_of_research, relevancy_pass_rate, num_search_urls, num_search_arxiv, progress=gr.Progress()):
    history = to_openai_format(message, history)
    table_of_concepts = None
    if len(table_of_concepts_json) > 0:
        table_of_concepts = TableOfConcepts.model_validate_json(table_of_concepts_json)
    try:
        result = await Runner.run(table_of_concepts_agent, history)
        table_of_concepts = TableOfConcepts.model_validate(result.final_output)
        history.append(EasyInputMessageParam(role="assistant", content="Подходит ли вам такое содержание? Что мне нужно поменять?\n\n" + table_of_concepts.print()))
        return message, to_gradio_format(history), table_of_concepts.model_dump_json()
    except InputGuardrailTripwireTriggered as e:
        with trace("Research workflow"):
            # Оглавление готово
            done_chapters = {}
            dic_visited_urls = defaultdict(list)
            progress_counts = 0
            for chapter in table_of_concepts.chapters:

                if chapter.need_research:
                    progress_counts += 1

                    summaries = []
                    hypos = []
                    visited_urls = []
                    context = {
                        'title': table_of_concepts.title,
                        'chapter_name': chapter.chapter_name,
                        'chapter_description': chapter.chapter_description,
                        'visited_urls': visited_urls,
                        'summaries': summaries,
                        'hypos': hypos
                    }
                    for depth in range(depth_of_research):
                        result = await Runner.run(follow_up_questions_agent, [], context=context)
                        result = FollowUpQuestions.model_validate(result.final_output)

                        for i, question in enumerate(result.questions):
                            if i < breadth_of_research:
                                progress(progress_counts/len(table_of_concepts.chapters), desc=f"Глава '{chapter.chapter_name}', ищем ответ на вопрос '{question}'. Вопрос {depth * breadth_of_research + i + 1} из {depth_of_research * breadth_of_research}")

                                web_search = await search_web(question, relevancy_pass_rate, num_search_urls, visited_urls)
                                if web_search is not None:
                                    summaries.append(web_search)

                                arxiv_search = await search_arxiv_relevant_pdfs_and_summarize(question, relevancy_pass_rate, num_search_arxiv, visited_urls)
                                if arxiv_search is not None:
                                    summaries.append(arxiv_search)

                        result = await Runner.run(hypos_agent, [], context=context)
                        result = NewHypothesis.model_validate(result.final_output)
                        hypos.extend(result.list_of_brilliant_ideas)

                    context['done_work'] = get_research(table_of_concepts, dic_visited_urls, done_chapters, final=False)
                    result = await Runner.run(chapter_editor_agent, [], context=context)
                    result = ChapterText.model_validate(result.final_output)
                    print("\n+++++\n", result)
                    done_chapters[chapter.chapter_name] = result.chapter_text_without_title_in_head
                    dic_visited_urls[chapter.chapter_name] = context['visited_urls']

            for chapter in table_of_concepts.chapters:
                if not chapter.need_research:
                    progress_counts += 1
                    progress(progress_counts/len(table_of_concepts.chapters), desc=f"Пишем главу {chapter.chapter_name}")
                    context = {
                        'title': table_of_concepts.title,
                        'chapter_name': chapter.chapter_name,
                        'chapter_description': chapter.chapter_description,
                        'done_chapters': done_chapters,
                        'done_work': get_research(table_of_concepts, dic_visited_urls, done_chapters, final=False)
                    }
                    result = await Runner.run(chapter_editor_summary_agent, [], context=context)
                    result = ChapterText.model_validate(result.final_output)
                    print("\n------\n", result)
                    done_chapters[chapter.chapter_name] = result.chapter_text_without_title_in_head

            final_research = get_research(table_of_concepts, dic_visited_urls, done_chapters, final=True)
            print("\n\n\n\n\n\n", final_research, '\n\n\n\n\n\n')
            history.append(EasyInputMessageParam(role="assistant", content=final_research))
            return "", to_gradio_format(history), table_of_concepts.model_dump_json()


def get_research(table_of_concepts, dic_visited_urls, done_chapters, final=False):
    text = f"# {table_of_concepts.title}\n"
    for chapter in table_of_concepts.chapters:
        if chapter.chapter_name in done_chapters:
            text = text + "\n" + f"# {chapter.chapter_name}\n{print_used_urls(dic_visited_urls[chapter.chapter_name]) if final else ''}\n{done_chapters[chapter.chapter_name]}"
    return text


with gr.Blocks() as app:
    with gr.Row(scale=5):
        with gr.Column(scale=5):
            chatbot = gr.Chatbot(type="messages", height='60vh', show_copy_button=True)
            msg = gr.Textbox(lines=5)
            table_of_concepts_box = gr.Textbox(visible=False)
            btn = gr.Button()
        with gr.Column(scale=1):
            breadth_of_research = gr.Slider(maximum=10, minimum=1, value=2, step=1, label='breadth_of_research')
            depth_of_research = gr.Slider(maximum=10, minimum=1, value=2, step=1, label='depth_of_research')
            relevancy_pass_rate = gr.Slider(maximum=10, minimum=1, value=7, step=1, label='relevancy_pass_rate')
            num_search_urls = gr.Slider(maximum=10, minimum=1, value=5, step=1, label='num_search_pages')
            num_search_arxiv = gr.Slider(maximum=10, minimum=1, value=3, step=1, label='num_search_arxiv')
        btn.click(chat,
                   [msg, chatbot, table_of_concepts_box, breadth_of_research, depth_of_research, relevancy_pass_rate, num_search_urls, num_search_arxiv],
                   [msg, chatbot, table_of_concepts_box],
                   show_progress_on=msg
                   )

app.launch()
