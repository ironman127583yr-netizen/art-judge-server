from fastapi import FastAPI
from pydantic import BaseModel
import psycopg2
import time
import asyncio

# ===============================
# CONFIG
# ===============================

DATABASE_URL = "YOUR_POSTGRES_URL"

# ===============================
# APP
# ===============================

app = FastAPI()
QUEUE = asyncio.Queue()

# ===============================
# DB
# ===============================

def get_conn():
    return psycopg2.connect(DATABASE_URL)

# ===============================
# MODELS
# ===============================

class InitRequest(BaseModel):
    matchId: str
    playerId: str
    duration: int = 30
    modeCategory: str = "Doodle"

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
            reference_index, duration, mode_category
        ) VALUES (%s,%s,'CREATED',%s,%s,%s)
        """, (
            data.matchId,
            data.playerId,
            ref_index,
            data.duration,
            data.modeCategory
        ))

        conn.commit()

        return {
            "referenceIndex": ref_index,
            "startTimestamp": 0,
            "duration": data.duration
        }

    # existing match
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

    if not start:
        return {"error": "not started"}

    if int(time.time()*1000) > start + duration*1000:
        return {"error": "time over"}

    if data.playerId not in [player_a, player_b]:
        return {"error": "invalid player"}

    if data.playerId == player_a:
        cur.execute("""
        UPDATE matches SET art_a=%s WHERE match_id=%s
        """, (data.artUrl, data.matchId))
    else:
        cur.execute("""
        UPDATE matches SET art_b=%s WHERE match_id=%s
        """, (data.artUrl, data.matchId))

    # check both submitted
    cur.execute("SELECT art_a, art_b FROM matches WHERE match_id=%s", (data.matchId,))
    artA, artB = cur.fetchone()

    if artA and artB:
        cur.execute("""
        UPDATE matches SET state='JUDGING' WHERE match_id=%s
        """, (data.matchId,))
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
# WORKER (DUMMY JUDGE)
# ===============================

async def judge_worker():
    while True:
        match_id = await QUEUE.get()

        conn = get_conn()
        cur = conn.cursor()

        # fake result for now
        result = {
            "winner": "playerA",
            "scoreA": 80,
            "scoreB": 70
        }

        cur.execute("""
        UPDATE matches
        SET state='FINISHED', result=%s
        WHERE match_id=%s
        """, (str(result), match_id))

        conn.commit()

# ===============================
# START
# ===============================

@app.on_event("startup")
async def startup():
    asyncio.create_task(judge_worker())
