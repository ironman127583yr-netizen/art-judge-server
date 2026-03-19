from fastapi import FastAPI
import asyncio

from worker import judge_worker
from match import router

app = FastAPI()

app.include_router(router)

@app.on_event("startup")
async def startup():
    asyncio.create_task(judge_worker())
