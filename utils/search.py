import httpx


async def search_news(query: str):
    # Пример использования Tavily API для поиска новостей
    api_key = "YOUR_TAVILY_API_KEY"
    url = f"https://tavily-search.p.rapidapi.com/search?q={query}&tool=web_search&rapidapi-key={api_key}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        # Парсим данные и возвращаем новости
        news = [item['title'] for item in data['results']]
        return news


async def search_links(query: str):
    # Пример использования Tavily API для поиска ссылок
    api_key = "YOUR_TAVILY_API_KEY"
    url = f"https://tavily-search.p.rapidapi.com/search?q={query}&tool=web_search&rapidapi-key={api_key}"

    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        data = response.json()
        # Парсим данные и возвращаем ссылки
        links = [item['url'] for item in data['results']]
        return links[:3]  # Возвращаем не более трех ссылок