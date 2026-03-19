from fastapi import APIRouter
import time
import random

from store import MATCHES
from worker import QUEUE

router = APIRouter()

# simple reference pool
REFERENCE_POOL = [
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/0.jpg",
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/1.jpg",
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/2.jpg",
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/3.jpg",
]


# =========================================================
# INITIALIZE MATCH
# =========================================================

@router.post("/initializeMatch")
async def initialize_match(data: dict):

    match_id = data["matchId"]
    player_id = data["playerId"]
    duration = data.get("duration", 30)

    match = MATCHES.get(match_id)

    if not match:
        ref_index = random.randint(0, len(REFERENCE_POOL) - 1)

        match = {
            "matchId": match_id,
            "playerA": player_id,
            "playerB": None,
            "artA": None,
            "artB": None,
            "state": "CREATED",
            "referenceIndex": ref_index,
            "referenceUrl": REFERENCE_POOL[ref_index],
            "startTimestamp": 0,
            "duration": duration,
            "result": None
        }

    elif not match["playerB"] and match["playerA"] != player_id:
        match["playerB"] = player_id

    # activate match
    if match["playerA"] and match["playerB"] and match["state"] == "CREATED":
        match["state"] = "ACTIVE"
        match["startTimestamp"] = int(time.time() * 1000)

    MATCHES[match_id] = match

    return {
        "referenceIndex": match["referenceIndex"],
        "startTimestamp": match["startTimestamp"],
        "duration": match["duration"]
    }


# =========================================================
# SUBMIT ART
# =========================================================

@router.post("/submitArt")
async def submit_art(data: dict):

    match = MATCHES.get(data["matchId"])

    if not match:
        return {"error": "match not found"}

    player_id = data["playerId"]

    if player_id == match["playerA"]:
        if match["artA"] is not None:
            return {"error": "already submitted"}
        match["artA"] = data["artUrl"]

    elif player_id == match["playerB"]:
        if match["artB"] is not None:
            return {"error": "already submitted"}
        match["artB"] = data["artUrl"]

    else:
        return {"error": "invalid player"}

    # trigger judge
    if match["artA"] and match["artB"] and match["state"] == "ACTIVE":
        match["state"] = "JUDGING"
        await QUEUE.put(match["matchId"])

    return {"status": "ok"}


# =========================================================
# GET MATCH STATE
# =========================================================

@router.get("/getMatchState")
async def get_match(matchId: str):

    match = MATCHES.get(matchId)

    if not match:
        return {"state": "UNKNOWN"}

    return {
        "state": match["state"],
        "result": match.get("result"),
        "startTimestamp": match.get("startTimestamp", 0),
        "duration": match.get("duration", 30)
    }
