import os
import ssl
import uuid
import httpx
from collections import defaultdict
from agents import set_default_openai_client, set_default_openai_api, set_trace_processors, Runner, trace, \
    OpenAIChatCompletionsModel
from agents.models import openai_provider
from gradio import ChatMessage
from openai import AsyncOpenAI
from openai.types.responses import EasyInputMessageParam
from phoenix.otel import register

from research_agents import TableOfConceptsAgent, FollowUpQuestionsAgent, \
    HyposGeneratingAgent, ChapterEditorAgent, ChapterEditorSummaryAgent, TableOfConceptsSearchAgent
from structured_outputs import TableOfConcepts, FollowUpQuestions, NewHypothesis, \
    ChapterText
from tools import search_web, search_arxiv_relevant_pdfs_and_summarize

ssl._create_default_https_context = ssl._create_unverified_context

PHOENIX_TRACE_URL = os.getenv("PHOENIX_TRACE_URL", "http://localhost:6006/v1/traces")
PHOENIX_PROJECT_NAME = os.getenv("PHOENIX_PROJECT_NAME", "deep-research")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY",  "1")
OPENAI_API_URL = os.getenv("OPENAI_API_URL", "https://openrouter.ai/api/v1")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "openai/gpt-4.1-mini")
TABLE_OF_CONCEPTS_MODEL = os.getenv("TABLE_OF_CONCEPTS_MODEL", "openai/gpt-4.1-mini")


default_model = OpenAIChatCompletionsModel(model=DEFAULT_MODEL, openai_client=AsyncOpenAI(base_url=OPENAI_API_URL, api_key=OPENAI_API_KEY, timeout=60 * 5, http_client=httpx.AsyncClient(verify=False)))
table_of_concepts_model = OpenAIChatCompletionsModel(model=TABLE_OF_CONCEPTS_MODEL, openai_client=AsyncOpenAI(base_url=OPENAI_API_URL, api_key=OPENAI_API_KEY, timeout=60 * 5, http_client=httpx.AsyncClient(verify=False)))

# configure the Phoenix tracer
set_trace_processors([])
tracer_provider = register(
    project_name=PHOENIX_PROJECT_NAME,  # Default is 'default'
    endpoint=PHOENIX_TRACE_URL,
    auto_instrument=True
)

set_default_openai_client(AsyncOpenAI(base_url=OPENAI_API_URL, api_key=OPENAI_API_KEY, timeout=60 * 5, http_client=httpx.AsyncClient(verify=False)))
set_default_openai_api('chat_completions')
openai_provider.DEFAULT_MODEL = DEFAULT_MODEL


table_of_concepts_agent = TableOfConceptsAgent(model=table_of_concepts_model)
table_of_concepts_search = TableOfConceptsSearchAgent(model=table_of_concepts_model)
follow_up_questions_agent = FollowUpQuestionsAgent(model=default_model)
hypos_agent = HyposGeneratingAgent(model=default_model)
chapter_editor_agent = ChapterEditorAgent(model=default_model)
chapter_editor_summary_agent = ChapterEditorSummaryAgent(model=default_model)


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



async def generate_table_of_concepts(message, history):
    history.append(EasyInputMessageParam(role='user', content=message))
    with trace("Table of concepts workflow", group_id=str(uuid.uuid4())):
        result = await Runner.run(table_of_concepts_search, history)
        result = await Runner.run(table_of_concepts_agent,  history + [EasyInputMessageParam(role="assistant", content=result.final_output), EasyInputMessageParam(role="user", content="перепиши в json")])
        table_of_concepts = TableOfConcepts.model_validate(result.final_output)
        history.append(EasyInputMessageParam(role="assistant", content="Хотите ли вы что-то поменять в структуре исследования?\n" + table_of_concepts.print()))
    return table_of_concepts



async def generate_research(table_of_concepts, history, breadth_of_research=3, depth_of_research=2, relevancy_pass_rate=9, num_search_urls=3, num_search_arxiv=2):
    with trace("Research workflow", group_id=str(uuid.uuid4())):
        # Оглавление готово
        done_chapters = {}
        dic_visited_urls = defaultdict(list)
        progress_counts = 0
        progress_len = breadth_of_research * depth_of_research * len([x for x in table_of_concepts.chapters if x.need_research]) + len([x for x in table_of_concepts.chapters if not x.need_research])
        for chapter in table_of_concepts.chapters:

            if chapter.need_research:
                summaries = []
                hypos = []
                visited_urls = set()
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
                            progress_counts += 1
                            try:
                                yield {
                                    "progress": f"Прогресс: {progress_counts/progress_len*100:.0f}%. Ищем ответ на вопрос '{question}'.",
                                    "research": "",
                                    "final": False
                                }
                                web_search = await search_web(question, relevancy_pass_rate, num_search_urls, visited_urls)
                                arxiv_search = await search_arxiv_relevant_pdfs_and_summarize(question, relevancy_pass_rate, num_search_arxiv, visited_urls)
                                if web_search is not None:
                                    summaries.append(web_search)

                                if arxiv_search is not None:
                                    summaries.append(arxiv_search)
                            except Exception as e:
                                print(f"Answering question: {str(e)}")

                    result = await Runner.run(hypos_agent, [], context=context)
                    result = NewHypothesis.model_validate(result.final_output)
                    hypos.extend(result.list_of_brilliant_ideas)

                context['done_work'] = get_research(table_of_concepts, dic_visited_urls, done_chapters, final=False)
                result = await Runner.run(chapter_editor_agent, [], context=context)
                result = ChapterText.model_validate(result.final_output)
                done_chapters[chapter.chapter_name] = result.chapter_text_without_title_in_head
                dic_visited_urls[chapter.chapter_name] = context['visited_urls']

        for chapter in table_of_concepts.chapters:
            if not chapter.need_research:
                progress_counts += 1
                yield {
                    "progress": f"Прогресс: {progress_counts/progress_len*100:.0f}%. Пишем главу {chapter.chapter_name}",
                    "research": "",
                    "final": False
                }
                context = {
                    'title': table_of_concepts.title,
                    'chapter_name': chapter.chapter_name,
                    'chapter_description': chapter.chapter_description,
                    'done_chapters': done_chapters,
                    'done_work': get_research(table_of_concepts, dic_visited_urls, done_chapters, final=False)
                }
                result = await Runner.run(chapter_editor_summary_agent, [], context=context)
                result = ChapterText.model_validate(result.final_output)
                done_chapters[chapter.chapter_name] = result.chapter_text_without_title_in_head

        final_research = get_research(table_of_concepts, dic_visited_urls, done_chapters, final=True)
        history.append(EasyInputMessageParam(role="assistant", content=final_research))
    yield {
        "progress": "Готово",
        "research": final_research,
        "final": True
    }


def get_research(table_of_concepts, dic_visited_urls, done_chapters, final=False):
    text = f"# {table_of_concepts.title}\n"
    for chapter in table_of_concepts.chapters:
        if chapter.chapter_name in done_chapters:
            text = text + "\n" + f"# {chapter.chapter_name}\n{print_used_urls(dic_visited_urls[chapter.chapter_name]) if final else ''}\n{done_chapters[chapter.chapter_name]}"
    return text
