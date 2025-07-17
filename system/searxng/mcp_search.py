from typing import List, Dict, Any

import requests
from fastmcp import FastMCP

mcp = FastMCP("SearXNG Search")


@mcp.tool
def searxng_search(query: str) -> List[Dict[str, Any]]:
    """
    Поиск в интернете через SearXNG для получения релевантных URL и информации.

    Этот инструмент выполняет поиск в интернете, используя SearXNG метапоисковик,
    который агрегирует результаты из множества поисковых систем (Google, DuckDuckGo,
    Brave и др.) для получения наиболее релевантных результатов.

    Args:
        query (str): Поисковый запрос. Может содержать:
            - Обычные ключевые слова (например: "машинное обучение")
            - Фразы в кавычках для точного поиска (например: "градиентный бустинг")
            - Запросы на любом языке (русский, английский и др.)

    Returns:
        List[Dict[str, Any]]: Список найденных результатов, где каждый элемент содержит:
            - url (str): URL найденной страницы
            - title (str): Заголовок страницы
            - content (str): Краткое описание/отрывок контента
            - score (float): Оценка релевантности результата (чем выше, тем лучше)
            - engine (str): Основной поисковый движок, который нашел результат
            - engines (List[str]): Все движки, которые нашли этот результат
            - positions (List[int]): Позиции в результатах разных движков
            - category (str): Категория результата (обычно "general")
            - publishedDate (str|None): Дата публикации (если доступна)
            - thumbnail (str|None): URL превью изображения (если доступно)
            - parsed_url (List[str]): Разобранный URL [protocol, domain, path, ...]

    Raises:
        requests.exceptions.RequestException: Если не удается подключиться к SearXNG
        requests.exceptions.HTTPError: Если SearXNG возвращает ошибку HTTP
        requests.exceptions.Timeout: Если запрос превышает таймаут (30 сек)

    Note:
        - Результаты сортируются по релевантности (score)
        - Максимальное время ожидания ответа: 30 секунд
        - Поддерживаются запросы на любом языке
        - Агрегирует результаты из множества поисковых движков
    """
    try:
        response = requests.get(
            f"http://localhost:8080/search?q={query}&format=json",
            timeout=30
        )
        response.raise_for_status()

        search_results = response.json()
        return search_results.get('results', [])

    except requests.exceptions.RequestException as e:
        # Логируем ошибку и возвращаем пустой список вместо падения
        print(f"Ошибка при выполнении поиска: {e}")
        return []


if __name__ == "__main__":
    mcp.run(transport="sse", host='0.0.0.0')