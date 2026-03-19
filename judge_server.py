from fastapi import FastAPI
import numpy as np
import cv2
from PIL import Image
import io
import requests
import asyncio
from pydantic import BaseModel

app = FastAPI()

# =========================================================
# IN-MEMORY STORE (TEMP — replace with DB later)
# =========================================================

MATCHES = {}
QUEUE = asyncio.Queue()

# =========================================================
# STARTUP
# =========================================================

@app.on_event("startup")
async def start_worker():
    asyncio.create_task(judge_worker())

# =========================================================
# IMAGE LOADING
# =========================================================

def load_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert("L")
    img = img.resize((256, 256))
    return np.array(img)

# =========================================================
# PROCESSING
# =========================================================

def build_structural_maps(img):
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    edges = cv2.Canny(blur, 70, 150)

    _, binary = cv2.threshold(blur, 40, 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    silhouette = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return {
        "edges": edges,
        "silhouette": silhouette
    }

def silhouette_similarity(ref, player):
    ref_bin = ref["silhouette"] > 0
    player_bin = player["silhouette"] > 0

    intersection = np.logical_and(ref_bin, player_bin)
    union = np.logical_or(ref_bin, player_bin)

    return float((intersection.sum() / (union.sum() + 1e-6)) * 100)

# =========================================================
# JUDGE CORE
# =========================================================

def judge_internal(reference_url, artA_url, artB_url):
    ref = build_structural_maps(load_image_from_url(reference_url))
    a = build_structural_maps(load_image_from_url(artA_url))
    b = build_structural_maps(load_image_from_url(artB_url))

    scoreA = silhouette_similarity(ref, a)
    scoreB = silhouette_similarity(ref, b)

    if scoreA > scoreB:
        winner = "playerA"
    elif scoreB > scoreA:
        winner = "playerB"
    else:
        winner = "draw"

    return {
        "winner": winner,
        "scoreA": round(scoreA, 2),
        "scoreB": round(scoreB, 2)
    }

# =========================================================
# WORKER (FIXED — NO SYNTAX ERROR)
# =========================================================

async def judge_worker():
    print("Judge worker started")

    while True:
        match_id = await QUEUE.get()

        match = MATCHES.get(match_id)
        if not match:
            continue

        try:
            result = judge_internal(
                match["referenceUrl"],
                match["artA"],
                match["artB"]
            )

            match["result"] = result
            match["state"] = "FINISHED"

            print(f"Match {match_id} finished")

        except Exception as e:
            print("Judge error:", e)

# =========================================================
# API
# =========================================================

@app.post("/initializeMatch")
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

    return {
        "state": match["state"]
    }

@app.post("/submitArt")
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

@app.get("/getMatchState")
async def get_match(matchId: str):
    match = MATCHES.get(matchId)

    if not match:
        return {"state": "UNKNOWN"}

    return match

# =========================================================
# TEST ENDPOINT
# =========================================================

class JudgeRequest(BaseModel):
    referenceUrl: str
    artAUrl: str
    artBUrl: str

@app.post("/judge")
async def judge(req: JudgeRequest):
    return judge_internal(
        req.referenceUrl,
        req.artAUrl,
        req.artBUrl
    )
