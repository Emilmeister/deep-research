import os

import requests
from agents import Runner, function_tool
from markdownify import markdownify
import re
from agents import Agent
from requests import RequestException

from structured_outputs import SummaryWithInterestingUrls, RelevanceScore, SearchWords, \
    TableOfConcepts
import arxiv

SEARXNG_SEARCH_URL = os.getenv("SEARXNG_SEARCH_URL", "http://localhost:8080/search")
PARSE_PDF_URL = os.getenv("PARSE_PDF_URL", "http://localhost:8000/extract-text")
MAX_CONTENT_LEN = int(os.getenv("MAX_CONTENT_LEN", 200000))

client = arxiv.Client()


@function_tool
async def search_web_tool(query: str) -> str:
    """ Используй для поиска в интернете. Тебе вернется саммари ответов релевантных веб страниц"""
    return await search_web(query, 7, 10, [])

@function_tool
async def final_answer_table_of_concepts(answer: TableOfConcepts) -> TableOfConcepts:
    """Используй в случае когда сделал предварительные поиски в интернете и хочешь написать итоговую структуру исследования"""
    return answer


async def search_web(query: str, relevancy_pass_rate: int, num_search: int, visited_urls: list[str]):
    """Используй для поиска инфорации в интернете
        Args:
        query: запрос

        Returns:
            Поисковая выдача
    """
    if num_search == 0:
        return

    results = searxng_search(keywords=query, max_results=num_search)
    summaries = []

    for result in results:
        summary = await visit_webpage_and_summarize(result['url'], query)
        if summary is not None and summary.relevance_score >= relevancy_pass_rate:
            visited_urls.append(result['url'])
            summaries.append(summary)

    all_interesting_urls = []

    for summary in summaries:
        all_interesting_urls.extend(
            [x.web_page_url for x in summary.interesting_web_page_urls if x.question_and_url_relevant_score >= relevancy_pass_rate])

    for url in all_interesting_urls:
        summary = await visit_webpage_and_summarize(url, query)
        if summary is not None and summary.relevance_score >= relevancy_pass_rate:
            visited_urls.append(url)
            summaries.append(summary)

    if list(summaries) == 0:
        return None

    return await summarize_texts(query, summaries)


async def visit_webpage_and_summarize(url: str, query: str):
    # """Посещение веб-страницы по URL и возвращение ее контента в markdown
    #
    # Args:
    #     url: URL веб-страницы, которую ты хочешь посетить.
    #
    # Returns:
    #     Контент веб-страницы в markdown, или ошибка если запрос выполнился некорректно
    # """

    if url.startswith("https://arxiv.org/abs/"):
        url = url.replace("abs", "pdf")

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        content = None

        if response.headers['content-type'] == 'application/pdf':
            content = await parse_pdf(url)
        else:
            markdown_content = markdownify(response.text).strip()
            content = re.sub(r"\n{3,}", "\n\n", markdown_content)

        if content is None:
            return

        return await summarize_content(query, url, content, "веб страница")
    except RequestException as e:
        print(f"Error fetching the webpage {url}: {str(e)}")
    except Exception as e:
        print(f"An unexpected error occurred {url}: {str(e)}")


def searxng_search(keywords, max_results):
    response = requests.get(f"{SEARXNG_SEARCH_URL}?q={keywords}&format=json", timeout=30)
    response.raise_for_status()
    return response.json()['results'][:max_results]


async def parse_pdf(url: str):
    try:
        response = requests.post(PARSE_PDF_URL, json={"url": url}, timeout=300)
        return response.json()['text']
    except Exception as e:
        print(f"An unexpected error occurred while parse_pdf {url}: {str(e)}")


async def search_arxiv_relevant_pdfs_and_summarize(question: str, relevancy_pass_rate: int, num_search: int, visited_urls: list[str]):
    question_to_words_agent = Agent(
        name="Questions to words agent",
        instructions=f"""
 Преобразуй следующий вопрос в набор ключевых слов НА АНГЛИЙСКОМ ЯЗЫКЕ, подходящих для поиска в поисковой системе.  

**Требования:**  
1. Удали все лишние слова (местоимения, предлоги, союзы, вводные конструкции).  
2. Оставь только значимые слова, отражающие суть запроса.  
3. Если есть конкретные параметры (даты, числа, названия), обязательно включи их.  
4. Слова должны быть в начальной форме (именительный падеж, инфинитив для глаголов).  
5. Не пиши словосочетания, разделяй их.  

**Примеры:**  
- Вопрос: "Как приготовить вкусные блины на молоке?" → Ключевые слова: ["recept", "yummy", "pancake", "milk"]`  
- Вопрос: "Где найти курсы по Python для начинающих в 2024 году?" → Ключевые слова: ["course", "python", "beginner", "2024"]  

**Входной вопрос:**  
{question}
            """,
        output_type=SearchWords
    )
    if num_search == 0:
        return

    agent_result = await Runner.run(question_to_words_agent, [])
    words = SearchWords.model_validate(agent_result.final_output).words
    articles = await search_arxiv_relevant_pdfs(words, question, num_search)
    summaries = []
    for article in articles:
        if article['relevance_score'] >= relevancy_pass_rate:
            content = await parse_pdf(article['pdf_url'])
            if content is not None:
                summary = await summarize_content(question, article['pdf_url'], content, "статья из научного журнала")
                if summary.relevance_score >= relevancy_pass_rate:
                    summaries.append(summary)
                    visited_urls.append(article['pdf_url'])

    return await summarize_texts(question, summaries)


async def search_arxiv_relevant_pdfs(search_words: list[str], question: str, max_results: int):
    words = ' '.join(search_words)
    search = arxiv.Search(
        query=f'ti:{words} AND abs:{words}',
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    results = []
    for result in client.results(search):
        article_relevance_agent = Agent(
            name="Article relevance agent",
            instructions=f"""
Ты — эксперт в анализе научных и публицистических текстов. Оцени, насколько аннотация статьи отвечает на **конкретный вопрос**, используя шкалу от 0 до 10. 

## Критерии оценки:

### 0 баллов:
- Аннотация не содержит информации, связанной с вопросом.
- Ключевые аспекты вопроса полностью игнорируются.
Пример: 
Вопрос: "Как ИИ улучшает диагностику рака?"  
Аннотация: "Исследование влияния диеты на уровень холестерина".

### 1-3 балла:
- Упоминаются смежные понятия, но прямой ответ на вопрос отсутствует.
- Нет данных, методов или выводов, релевантных вопросу.
Пример: 
Вопрос: "Каковы риски блокчейна для банков?"  
Аннотация: "Обзор технологий распределенного реестра".

### 4-5 баллов:
- Есть косвенная связь с вопросом, но ответ неполный или поверхностный.
- Затрагивается лишь часть вопроса без детализации.
Пример: 
Вопрос: "Почему нейросети хуже распознают редкие заболевания?"  
Аннотация: "Рассмотрены ограничения ИИ в медицине".

### 6-7 баллов:
- Аннотация частично отвечает на вопрос, но с пробелами.
- Указаны некоторые релевантные данные, но без глубины анализа.
Пример: 
Вопрос: "Как климат влияет на миграцию птиц в Европе?"  
Аннотация: "Исследование сезонной миграции воробьиных (без привязки к климату)".

### 8-9 баллов:
- Четкий ответ на вопрос с аргументацией или данными.
- Не хватает лишь небольших уточнений или примеров.
Пример: 
Вопрос: "Какие алгоритмы машинного обучения эффективны для прогнозирования цен на нефть?"  
Аннотация: "Сравнение LSTM и Random Forest для прогнозирования цен на нефть (точность LSTM — 89%)".

### 10 баллов:
- Полный и исчерпывающий ответ на вопрос.
- Включены: методы, результаты, выводы и значимость именно для заданного вопроса.
Пример: 
Вопрос: "Как социальные сети влияют на тревожность у подростков?"  
Аннотация: "Лонгитюдное исследование 1000 подростков показало, что ежедневное использование соцсетей >3 часов увеличивает тревожность на 40% (p < 0.01), особенно у девушек".

## Запрос на оценку:
**Вопрос:** {question}
**Аннотация:** {result.summary}
            """,
            output_type=RelevanceScore
        )
        agent_result = await Runner.run(article_relevance_agent, [])

        results.append({
            'title': result.title,
            'pdf_url': result.pdf_url,
            'abstract': result.summary,
            'relevance_score': RelevanceScore.model_validate(agent_result.final_output).relevance_score
        })
    return results


async def summarize_content(query: str, url: str, content: str, source: str) -> SummaryWithInterestingUrls:
    if len(content) > MAX_CONTENT_LEN:
        content = content[:MAX_CONTENT_LEN]

    search_summary_agent = Agent(
        name="Web page summary agent",
        instructions=f"""
Суммаризируй контент c "{source}" (url={url}) с учетом того что нас интересует ответ на следующий вопрос.
ВОПРОС: {query}
Так же ответь на вопрос, релевантен ли контент веб страницы этому ВОПРОСУ по шкале от 0 до 10, а ответ запиши в поле relevance_score.  
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
{content}
            """,
        output_type=SummaryWithInterestingUrls
    )
    result = await Runner.run(search_summary_agent, [])
    result = SummaryWithInterestingUrls.model_validate(result.final_output)
    if result.interesting_web_page_urls is None:
        result.interesting_web_page_urls = []
    return result


async def summarize_texts(query: str, summaries: list[SummaryWithInterestingUrls]):
    if len(summaries) == 0:
        return None

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
