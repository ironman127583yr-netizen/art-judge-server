from fastapi import FastAPI
from pydantic import BaseModel
import time
import asyncio
import os
import psycopg2

from db import get_conn
from worker import QUEUE, judge_worker

app = FastAPI()

@app.get("/create-table")
def create_table():
    try:
        conn = psycopg2.connect(os.environ.get("DATABASE_URL"))
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
            result TEXT
        );
        """)

        conn.commit()
        cur.close()
        conn.close()

        return {"status": "table created or already exists"}

    except Exception as e:
        return {"error": str(e)}

# ===============================
# MODELS
# ===============================

class InitRequest(BaseModel):
    matchId: str
    playerId: str
    duration: int = 30

class SubmitRequest(BaseModel):
    matchId: str
    playerId: str
    artUrl: str

# ===============================
# INIT MATCH
# ===============================

@app.post("/initializeMatch")
async def initialize_match(data: InitRequest):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM matches WHERE match_id=%s", (data.matchId,))
    row = cur.fetchone()

    if not row:
        ref_index = abs(hash(data.matchId)) % 4

        cur.execute("""
        INSERT INTO matches (
            match_id, player_a, state,
            reference_index, duration
        ) VALUES (%s,%s,'CREATED',%s,%s)
        """, (
            data.matchId,
            data.playerId,
            ref_index,
            data.duration
        ))

        conn.commit()

        return {
            "referenceIndex": ref_index,
            "startTimestamp": 0,
            "duration": data.duration
        }

    player_a, player_b, state = row[1], row[2], row[5]

    if not player_b and player_a != data.playerId:
        cur.execute("""
        UPDATE matches SET player_b=%s WHERE match_id=%s
        """, (data.playerId, data.matchId))
        player_b = data.playerId

    if player_a and player_b and state == "CREATED":
        start = int(time.time() * 1000)

        cur.execute("""
        UPDATE matches
        SET state='ACTIVE', start_timestamp=%s
        WHERE match_id=%s
        """, (start, data.matchId))

        conn.commit()

        return {
            "referenceIndex": row[6],
            "startTimestamp": start,
            "duration": row[8]
        }

    conn.commit()

    return {
        "referenceIndex": row[6],
        "startTimestamp": row[7] or 0,
        "duration": row[8]
    }

# ===============================
# SUBMIT ART
# ===============================

@app.post("/submitArt")
async def submit_art(data: SubmitRequest):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM matches WHERE match_id=%s", (data.matchId,))
    match = cur.fetchone()

    if not match:
        return {"error": "match not found"}

    player_a, player_b = match[1], match[2]
    state = match[5]
    start = match[7]
    duration = match[8]

    if state != "ACTIVE":
        return {"error": "not active"}

    if int(time.time()*1000) > start + duration*1000:
        return {"error": "time over"}

    if data.playerId not in [player_a, player_b]:
        return {"error": "invalid player"}

    if data.playerId == player_a:
        cur.execute("UPDATE matches SET art_a=%s WHERE match_id=%s",
                    (data.artUrl, data.matchId))
    else:
        cur.execute("UPDATE matches SET art_b=%s WHERE match_id=%s",
                    (data.artUrl, data.matchId))

    # check both submitted
    cur.execute("SELECT art_a, art_b FROM matches WHERE match_id=%s", (data.matchId,))
    artA, artB = cur.fetchone()

    if artA and artB:
        cur.execute("UPDATE matches SET state='JUDGING' WHERE match_id=%s",
                    (data.matchId,))
        await QUEUE.put(data.matchId)

    conn.commit()

    return {"status": "ok"}

# ===============================
# GET STATE
# ===============================

@app.get("/getMatchState")
async def get_match(matchId: str):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM matches WHERE match_id=%s", (matchId,))
    match = cur.fetchone()

    if not match:
        return {"state": "UNKNOWN"}

    return {
        "state": match[5],
        "startTimestamp": match[7] or 0,
        "duration": match[8],
        "result": match[9]
    }

# ===============================
# START WORKER
# ===============================

@app.on_event("startup")
async def startup():
    asyncio.create_task(judge_worker())
