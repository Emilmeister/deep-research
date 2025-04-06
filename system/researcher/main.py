import os
import ssl
import uuid
from collections import defaultdict

import gradio as gr
from agents import set_default_openai_client, set_default_openai_api, set_trace_processors, Runner, input_guardrail, \
    GuardrailFunctionOutput, trace
from agents.models import openai_provider
from gradio import ChatMessage
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
from phoenix.otel import register

from research_agents import TableOfConceptsAgent, TableOfConceptsGuardrailAgent, FollowUpQuestionsAgent, \
    HyposGeneratingAgent, ChapterEditorAgent, ChapterEditorSummaryAgent, TableOfConceptsSearchAgent
from structured_outputs import TableOfConcepts, TableOfConceptsGuardrail, FollowUpQuestions, NewHypothesis, \
    ChapterText
from tools import search_web, search_arxiv_relevant_pdfs_and_summarize

ssl._create_default_https_context = ssl._create_unverified_context

PHOENIX_TRACE_URL = os.getenv("PHOENIX_TRACE_URL", "http://localhost:6006/v1/traces")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "deep-research")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY",  "1")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "http://localhost:11434/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:32b")

# configure the Phoenix tracer
set_trace_processors([])
tracer_provider = register(
    project_name=PHOENIX_PROJECT_NAME,  # Default is 'default'
    endpoint=PHOENIX_TRACE_URL,
    auto_instrument=True
)

set_default_openai_client(AsyncOpenAI(base_url=OPENAI_API_URL, api_key=OPENAI_API_KEY, timeout=60 * 5))
set_default_openai_api('chat_completions')
openai_provider.DEFAULT_MODEL = DEFAULT_MODEL


table_of_concepts_agent = TableOfConceptsAgent()
table_of_concepts_search = TableOfConceptsSearchAgent()
follow_up_questions_agent = FollowUpQuestionsAgent()
hypos_agent = HyposGeneratingAgent()
chapter_editor_agent = ChapterEditorAgent()
chapter_editor_summary_agent = ChapterEditorSummaryAgent()


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


async def chat(message, start_research, history, table_of_concepts_json, breadth_of_research, depth_of_research, relevancy_pass_rate, num_search_urls, num_search_arxiv, progress=gr.Progress()):
    history = to_openai_format(message, history)
    table_of_concepts = None
    if len(table_of_concepts_json) > 0:
        table_of_concepts = TableOfConcepts.model_validate_json(table_of_concepts_json)
    if not start_research:
        with trace("Table of concepts workflow", group_id=str(uuid.uuid4())):
            result = await Runner.run(table_of_concepts_search, history)
            result = await Runner.run(table_of_concepts_agent,  history + [EasyInputMessageParam(role="assistant", content=result.final_output), EasyInputMessageParam(role="user", content="перепиши в json")])
            table_of_concepts = TableOfConcepts.model_validate(result.final_output)
            history.append(EasyInputMessageParam(role="assistant", content="Подходит ли вам такое содержание? Что мне нужно поменять?\n\n" + table_of_concepts.print()))
        return message, start_research, to_gradio_format(history), table_of_concepts.model_dump_json()
    else:
        with trace("Research workflow", group_id=str(uuid.uuid4())):
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
        return "", start_research, to_gradio_format(history), table_of_concepts.model_dump_json()


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
            with gr.Row(scale=5):
                btn = gr.Button()
                checkbox = gr.Checkbox(value=False, label="Начать исследование")
        with gr.Column(scale=1):
            breadth_of_research = gr.Slider(maximum=10, minimum=1, value=2, step=1, label='Количество вопросов для поиска в интернете в рамках главы')
            depth_of_research = gr.Slider(maximum=10, minimum=1, value=2, step=1, label='Количество циклов генерирования дополнительных вопросов и гипотез в рамках главы')
            relevancy_pass_rate = gr.Slider(maximum=10, minimum=0, value=7, step=1, label='Порог релевантности при выборе решении о переходе на другие страницы')
            num_search_urls = gr.Slider(maximum=10, minimum=0, value=5, step=1, label='Количество анализируемых страниц при поисковой выдаче')
            num_search_arxiv = gr.Slider(maximum=10, minimum=0, value=3, step=1, label='Количество анализируемых страниц при поиске в arxiv')
        btn.click(chat,
                   [msg, checkbox, chatbot, table_of_concepts_box, breadth_of_research, depth_of_research, relevancy_pass_rate, num_search_urls, num_search_arxiv],
                   [msg, checkbox, chatbot, table_of_concepts_box],
                   show_progress_on=msg
                   )

app.launch(server_name="0.0.0.0")
