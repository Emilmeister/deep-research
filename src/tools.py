import requests
from agents import function_tool, Runner
from duckduckgo_search import DDGS
from markdownify import markdownify
import re
from agents import Agent

# todo
from pypdf import PdfReader
from requests import RequestException

from src.model.structured_outputs import SummaryWithInterestingUrls

ddgs = DDGS()


# @function_tool
async def search_duckduckgo(query: str):
    """Используй для поиска инфорации в интернете
        Args:
        query: запрос

        Returns:
            Поисковая выдача
    """

    results = ddgs.text(keywords=query, max_results=2, region="ru-ru")
    summaries = []

    for result in results:
        summary = await visit_webpage_and_summarize(result['href'], query)
        if summary is not None and summary.is_this_page_relevant_to_question:
            summaries.append(summary)


    all_interesting_urls = []

    for summary in summaries:
        all_interesting_urls.extend([x.url for x in summary.interesting_urls if x.question_and_url_relevant_score > 7])

    summaries.extend([summary for url in all_interesting_urls if (summary := await visit_webpage_and_summarize(url, query)) is not None])

    n = '\n\n'
    summary_agent = Agent(
        name="Summary agent",
        instructions=f"""
                Сделай выжимку знаний из текстов с учетом того что нас интересует ответ на следующий вопрос.
                Вопрос: {query}
                
                !Не упоминай рекламу и нерелевантную информацию!
                
                Тексты:
                {n.join([summary.summary for summary in summaries])}
            """,
    )
    result = await Runner.run(summary_agent, [])
    return result.final_output


async def visit_webpage_and_summarize(url: str, query: str) -> SummaryWithInterestingUrls:
    # """Посещение веб-страницы по URL и возвращение ее контента в markdown
    #
    # Args:
    #     url: URL веб-страницы, которую ты хочешь посетить.
    #
    # Returns:
    #     Контент веб-страницы в markdown, или ошибка если запрос выполнился некорректно
    # """
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        markdown_content = markdownify(response.text).strip()
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        search_summary_agent = Agent(
            name="Web page summary agent",
            instructions=f"""
                Суммаризируй контент c веб страницы (url={url}) с учетом того что нас интересует ответ на следующий вопрос.
                ВОПРОС: {query}
                Так же ответь на вопрос, релевантен ли контент веб страницы этому ВОПРОСУ, а ответ запиши в поле is_this_page_relevant_to_question.  
                И если вдруг ты посчитаешь что есть большая вероятность найти ответ на этот ВОПРОС по какой либо другой ссылке,
                то напиши эту ссылку и объяснение почему ты так считаешь в массив interesting_urls (указывай в него только полные ссылки формата http://test.com/test... или https://test.com/test...). 
                В поле question_and_url_relevant_score напиши значение от 0 до 10 насколько ссылка релевантна ВОПРОСУ. 
                (
                    Критерии оценки is_this_page_relevant_to_question:
                    Релевантность анкора и контекста (0-6 балла)
                    0: Анкор и текст вокруг ссылки не связаны с вопросом.
                    1: Есть слабая связь по общей теме.
                    2: Анкор частично соответствует вопросу.
                    3: Прямое совпадение ключевых слов или явный намёк на ответ.
                    
                    Авторитетность источника (0-2 балла)
                    0: Сомнительный/неизвестный сайт.
                    1: Средняя надёжность (блоги, Wikipedia).
                    2: Экспертный источник (научные статьи, официальные сайты).
                    
                    Качество URL (0-2 балла)
                    0: Подозрительный URL (редиректы, clickbait).
                    1: Обычный URL без явных проблем.
                    2: Чистый, понятный URL (например, gov/edu-сайты).
                )
                
                !Не упоминай рекламу и нерелевантную информацию!
                
                Контент веб страницы:
                {markdown_content}
            """,
            output_type=SummaryWithInterestingUrls
        )
        result = await Runner.run(search_summary_agent, [])
        return SummaryWithInterestingUrls.model_validate(result.final_output)
    except RequestException as e:
        print(f"Error fetching the webpage {url}: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred {url}: {str(e)}")
