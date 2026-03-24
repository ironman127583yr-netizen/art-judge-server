from fastapi import FastAPI
from pydantic import BaseModel
import time
import asyncio
import os
import psycopg2

from db import get_conn
from worker import QUEUE, judge_worker

app = FastAPI()

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
# CREATE TABLE
# ===============================

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

        return {"status": "ok"}

    except Exception as e:
        return {"error": str(e)}


# ===============================
# INIT MATCH
# ===============================

@app.post("/initializeMatch")
async def initialize_match(data: InitRequest):

    conn = get_conn()
    cur = conn.cursor()

    try:
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

    except psycopg2.errors.UniqueViolation:
        conn.rollback()

    # fetch existing
    cur.execute("SELECT * FROM matches WHERE match_id=%s", (data.matchId,))
    row = cur.fetchone()

    if not row:
        return {"error": "match not found"}

    player_a, player_b, state = row[1], row[2], row[5]

    # join second player
    if not player_b and player_a != data.playerId:
        cur.execute(
            "UPDATE matches SET player_b=%s WHERE match_id=%s",
            (data.playerId, data.matchId)
        )
        conn.commit()

    # start match
    cur.execute("""
    UPDATE matches
    SET state='ACTIVE', start_timestamp=%s
    WHERE match_id=%s AND state='CREATED'
    RETURNING start_timestamp
    """, (int(time.time()*1000), data.matchId))

    started = cur.fetchone()

    if started:
        return {
            "referenceIndex": row[6],
            "startTimestamp": started[0],
            "duration": row[8]
        }

    return {
        "referenceIndex": row[6],
        "startTimestamp": row[7] or 0,
        "duration": row[8]
    }


# ===============================
# GET MATCH STATE
# ===============================

@app.get("/getMatchState")
def get_match_state(matchId: str):

    conn = get_conn()
    cur = conn.cursor()

    cur.execute("SELECT * FROM matches WHERE match_id=%s", (matchId,))
    row = cur.fetchone()

    if not row:
        return {"error": "match not found"}

    return {
        "state": row[5],
        "referenceIndex": row[6],
        "startTimestamp": row[7] or 0,
        "duration": row[8],
        "result": row[9]
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
    art_a, art_b = match[3], match[4]
    state = match[5]
    start = match[7]
    duration = match[8]

    if state != "ACTIVE":
        return {"error": "not active"}

    if not start:
        return {"error": "not started"}

    if int(time.time()*1000) > start + duration*1000:
        return {"error": "time over"}

    if data.playerId not in [player_a, player_b]:
        return {"error": "invalid player"}

    if data.playerId == player_a and art_a:
        return {"error": "already submitted"}

    if data.playerId == player_b and art_b:
        return {"error": "already submitted"}

    if data.playerId == player_a:
        cur.execute(
            "UPDATE matches SET art_a=%s WHERE match_id=%s",
            (data.artUrl, data.matchId)
        )
    else:
        cur.execute(
            "UPDATE matches SET art_b=%s WHERE match_id=%s",
            (data.artUrl, data.matchId)
        )

    conn.commit()

    cur.execute(
        "SELECT art_a, art_b, state FROM matches WHERE match_id=%s",
        (data.matchId,)
    )
    artA, artB, current_state = cur.fetchone()

    if artA and artB and current_state == "ACTIVE":
        cur.execute("""
        UPDATE matches
        SET state='JUDGING'
        WHERE match_id=%s AND state='ACTIVE'
        """, (data.matchId,))
        conn.commit()

        await QUEUE.put(data.matchId)

    return {"status": "ok"}


# ===============================
# START WORKER
# ===============================

@app.on_event("startup")
async def startup():
    asyncio.create_task(judge_worker())
