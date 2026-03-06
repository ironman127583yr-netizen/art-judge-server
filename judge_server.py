from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from PIL import Image
import io
import os
import uvicorn

app = FastAPI()


# =========================================================
# ROOT ENDPOINT
# =========================================================

@app.get("/")
def home():
    return {"status": "Art Judge Server Running"}


# =========================================================
# IMAGE LOADING
# =========================================================

def load_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    image = image.resize((256,256))
    return np.array(image)


# =========================================================
# PREPROCESSING
# =========================================================

def preprocess_reference(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    edges = cv2.Canny(blur,80,160)

    _, thresh = cv2.threshold(edges,40,255,cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    silhouette = cv2.dilate(thresh,kernel,iterations=1)

    return silhouette


def preprocess_player(img):
    blur = cv2.GaussianBlur(img,(3,3),0)
    edges = cv2.Canny(blur,60,140)
    return edges


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def edge_density(img):
    edges = cv2.Canny(img,100,200)
    return np.mean(edges)


def value_distribution(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    return np.std(hist)


def composition_balance(img):
    moments = cv2.moments(img)

    if moments["m00"] == 0:
        return 0

    cx = moments["m10"]/moments["m00"]
    cy = moments["m01"]/moments["m00"]

    return cx + cy


def detail_strength(img):
    lap = cv2.Laplacian(img,cv2.CV_64F)
    return lap.var()


def proportion_measure(img):

    contours,_ = cv2.findContours(
        cv2.Canny(img,100,200),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        return 0

    areas = [cv2.contourArea(c) for c in contours]

    return np.mean(areas)


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(player_feature, ref_feature):

    if ref_feature <= 1e-6:
        return 0

    ratio = player_feature / ref_feature
    score = ratio * 100

    return max(0,min(120,score))


# =========================================================
# METRIC COMPUTATION
# =========================================================

def compute_metrics(reference, player):

    ref_line = edge_density(reference)
    ref_value = value_distribution(reference)
    ref_comp = composition_balance(reference)
    ref_detail = detail_strength(reference)
    ref_prop = proportion_measure(reference)

    metrics = {

        "proportion": normalize(proportion_measure(player),ref_prop),

        "line": normalize(edge_density(player),ref_line),

        "value": normalize(value_distribution(player),ref_value),

        "composition": normalize(composition_balance(player),ref_comp),

        "detail": normalize(detail_strength(player),ref_detail)
    }

    return metrics


# =========================================================
# FINAL SCORE
# =========================================================

def calculate_score(metrics,weights):

    score = 0

    for key in metrics:
        score += metrics[key] * weights[key]

    return score


# =========================================================
# JUDGE ENDPOINT
# =========================================================

@app.post("/judge")
async def judge(
    reference: UploadFile = File(...),
    playerA: UploadFile = File(...),
    playerB: UploadFile = File(...)
):

    # load images
    ref_raw = load_image(await reference.read())
    a_raw = load_image(await playerA.read())
    b_raw = load_image(await playerB.read())

    # preprocess images
    ref = preprocess_reference(ref_raw)
    a = preprocess_player(a_raw)
    b = preprocess_player(b_raw)

    # compute metrics
    metricsA = compute_metrics(ref,a)
    metricsB = compute_metrics(ref,b)

    # scoring weights
    weights = {
        "proportion":0.30,
        "line":0.15,
        "value":0.25,
        "composition":0.10,
        "detail":0.20
    }

    scoreA = calculate_score(metricsA,weights)
    scoreB = calculate_score(metricsB,weights)

    winner = "draw"

    if scoreA > scoreB:
        winner = "playerA"

    elif scoreB > scoreA:
        winner = "playerB"


    return {

        "winner": winner,

        "scoreA": round(scoreA,2),
        "scoreB": round(scoreB,2),

        "weights": weights,

        "metricsA": metricsA,
        "metricsB": metricsB
    }
