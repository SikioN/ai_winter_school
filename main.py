import time
from typing import List
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import HttpUrl
from schemas.request import PredictionRequest, PredictionResponse
from utils.logger import setup_logger
from langchain.llms import OpenAI  # Используем стандартную библиотеку langchain
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from utils.search import search_news, search_links

# Initialize
app = FastAPI()
logger = None
llm = None  # Языковая модель


@app.on_event("startup")
async def startup_event():
    global logger, llm
    logger = await setup_logger()

    # Инициализация языковой модели
    llm = OpenAI(model_name="text-davinci-003", temperature=0)  # Пример использования OpenAI


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    body = await request.body()
    await logger.info(
        f"Incoming request: {request.method} {request.url}\n"
        f"Request body: {body.decode()}"
    )
    response = await call_next(request)
    process_time = time.time() - start_time
    response_body = b""
    async for chunk in response.body_iterator:
        response_body += chunk
    await logger.info(
        f"Request completed: {request.method} {request.url}\n"
        f"Status: {response.status_code}\n"
        f"Response body: {response_body.decode()}\n"
        f"Duration: {process_time:.3f}s"
    )
    return Response(
        content=response_body,
        status_code=response.status_code,
        headers=dict(response.headers),
        media_type=response.media_type,
    )


@app.post("/api/request", response_model=PredictionResponse)
async def predict(body: PredictionRequest):
    try:
        await logger.info(f"Processing prediction request with id: {body.id}")

        # Создаем шаблон для промпта
        prompt_template = "Ответь на вопрос: {query}"
        prompt = PromptTemplate(template=prompt_template, input_variables=["query"])

        # Создаем цепочку для обработки запроса
        chain = LLMChain(llm=llm, prompt=prompt)

        # Генерируем ответ с помощью языковой модели
        model_response = chain.run(query=body.query)

        # Парсим ответ
        answer, reasoning, sources = parse_model_response(model_response)

        # Поиск дополнительных ссылок
        additional_sources = await search_links(body.query)
        sources.extend(additional_sources)

        response = PredictionResponse(
            id=body.id,
            answer=answer,
            reasoning=reasoning,
            sources=sources,
        )
        await logger.info(f"Successfully processed request {body.id}")
        return response
    except ValueError as e:
        error_msg = str(e)
        await logger.error(f"Validation error for request {body.id}: {error_msg}")
        raise HTTPException(status_code=400, detail=error_msg)
    except Exception as e:
        await logger.error(f"Internal error processing request {body.id}: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


def parse_model_response(model_response: str):
    # Пример парсинга ответа от модели
    # В реальном случае это может быть более сложный процесс
    answer = 1  # Замените на реальный вызов модели
    reasoning = "Из информации на сайте"
    sources: List[HttpUrl] = [
        HttpUrl("https://itmo.ru/ru/"),
        HttpUrl("https://abit.itmo.ru/"),
    ]

    # Здесь нужно добавить логику для извлечения правильного ответа, рассуждений и источников
    # Например, можно использовать регулярные выражения или другие методы парсинга

    return answer, reasoning, sources


# Если нужно запустить локально без Docker (например, для тестирования)
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080, reload=True)