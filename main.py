from fastapi import FastAPI
from pydantic import BaseModel
import asyncio
import time
import random

from worker import judge_worker, QUEUE, MATCHES

app = FastAPI()
@app.get("/debug")
def debug():
    return {
        "status": "THIS IS NEW CODE",
        "version": "v2-clean"
    }

# =========================
# STARTUP
# =========================
@app.on_event("startup")
async def startup():
    asyncio.create_task(judge_worker())

# =========================
# DATA
# =========================
REFERENCE_POOL = [
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/0.jpg",
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/1.jpg",
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/2.jpg",
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/3.jpg",
]

# =========================
# REQUEST MODELS
# =========================
class InitRequest(BaseModel):
    matchId: str
    playerId: str
    duration: int = 30

class SubmitRequest(BaseModel):
    matchId: str
    playerId: str
    artUrl: str

# =========================
# INITIALIZE MATCH
# =========================
@app.post("/initializeMatch")
async def initialize_match(data: InitRequest):

    match = MATCHES.get(data.matchId)

    if match is None:
        ref_index = random.randint(0, len(REFERENCE_POOL) - 1)

        match = {
            "matchId": data.matchId,
            "playerA": data.playerId,
            "playerB": None,
            "artA": None,
            "artB": None,
            "state": "CREATED",
            "referenceIndex": ref_index,
            "referenceUrl": REFERENCE_POOL[ref_index],
            "startTimestamp": 0,
            "duration": data.duration,
            "result": None
        }

    else:
        if match["playerB"] is None and match["playerA"] != data.playerId:
            match["playerB"] = data.playerId

    # activate
    if match["playerA"] and match["playerB"] and match["state"] == "CREATED":
        match["state"] = "ACTIVE"
        match["startTimestamp"] = int(time.time() * 1000)

    MATCHES[data.matchId] = match

    return {
        "referenceIndex": match["referenceIndex"],
        "startTimestamp": match["startTimestamp"],
        "duration": match["duration"]
    }

# =========================
# SUBMIT ART
# =========================
@app.post("/submitArt")
async def submit_art(data: SubmitRequest):

    match = MATCHES.get(data.matchId)

    if not match:
        return {"error": "match not found"}

    if data.playerId == match["playerA"]:
        if match["artA"]:
            return {"error": "already submitted"}
        match["artA"] = data.artUrl

    elif data.playerId == match["playerB"]:
        if match["artB"]:
            return {"error": "already submitted"}
        match["artB"] = data.artUrl

    else:
        return {"error": "invalid player"}

    if match["artA"] and match["artB"]:
        match["state"] = "JUDGING"
        await QUEUE.put(match["matchId"])

    return {"status": "ok"}

# =========================
# GET MATCH STATE
# =========================
@app.get("/getMatchState")
async def get_match(matchId: str):

    match = MATCHES.get(matchId)

    if not match:
        return {"state": "UNKNOWN"}

    return {
        "state": match["state"],
        "result": match["result"],
        "startTimestamp": match["startTimestamp"],
        "duration": match["duration"]
    }
