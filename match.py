from fastapi import APIRouter
from store import MATCHES
from worker import QUEUE

router = APIRouter()


@router.post("/initializeMatch")
async def initialize_match(data: dict):

    match_id = data["matchId"]
    player_id = data["playerId"]

    match = MATCHES.get(match_id)

    if not match:
        match = {
            "matchId": match_id,
            "playerA": player_id,
            "playerB": None,
            "artA": None,
            "artB": None,
            "state": "CREATED",
            "referenceUrl": data["referenceUrl"]
        }

    elif not match["playerB"]:
        match["playerB"] = player_id

    if match["playerA"] and match["playerB"]:
        match["state"] = "ACTIVE"

    MATCHES[match_id] = match

    return match


@router.post("/submitArt")
async def submit_art(data: dict):

    match = MATCHES.get(data["matchId"])

    if not match:
        return {"error": "match not found"}

    if data["playerId"] == match["playerA"]:
        match["artA"] = data["artUrl"]
    else:
        match["artB"] = data["artUrl"]

    if match["artA"] and match["artB"]:
        match["state"] = "JUDGING"
        await QUEUE.put(match["matchId"])

    return {"status": "ok"}


@router.get("/getMatchState")
async def get_match(matchId: str):

    match = MATCHES.get(matchId)

    if not match:
        return {"state": "UNKNOWN"}

    return match
