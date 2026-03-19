from fastapi import FastAPI
import asyncio

from match import router
from worker import judge_worker

app = FastAPI()

# register routes
app.include_router(router)

# start background worker
@app.on_event("startup")
async def startup():
    asyncio.create_task(judge_worker())
