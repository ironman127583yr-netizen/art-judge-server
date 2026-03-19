from fastapi import FastAPI
import asyncio

from match import router
from worker import judge_worker

from db import get_conn
import asyncio

app = FastAPI()

# register routes
app.include_router(router)

# start background worker
@app.on_event("startup")
async def startup():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS matches (
        match_id TEXT PRIMARY KEY,
        player_a TEXT,
        player_b TEXT,
        art_a TEXT,
        art_b TEXT,
        state TEXT,
        reference_index INT,
        start_timestamp BIGINT,
        duration INT,
        result JSONB
    );
    """)

    cur.close()

    asyncio.create_task(judge_worker())
