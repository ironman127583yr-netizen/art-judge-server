from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()


@app.get("/")
def home():
    return {"status": "Art Judge Server Running"}


# -----------------------------
# Image Loader
# -----------------------------
def load_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    image = image.resize((128, 128))
    return np.array(image)


# -----------------------------
# Metric 1 : Proportion
# -----------------------------
def proportion_score(ref, img):
    diff = np.abs(ref.astype("float") - img.astype("float"))
    score = 100 - (np.mean(diff) / 255) * 100
    return np.clip(score, 0, 100)


# -----------------------------
# Metric 2 : Line Quality
# -----------------------------
def line_score(ref, img):
    ref_edges = cv2.Canny(ref, 100, 200)
    img_edges = cv2.Canny(img, 100, 200)

    diff = np.abs(ref_edges.astype("float") - img_edges.astype("float"))
    score = 100 - (np.mean(diff) / 255) * 100
    return np.clip(score, 0, 100)


# -----------------------------
# Metric 3 : Value Balance
# -----------------------------
def value_score(ref, img):
    ref_hist = cv2.calcHist([ref], [0], None, [256], [0, 256])
    img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    similarity = cv2.compareHist(ref_hist, img_hist, cv2.HISTCMP_CORREL)
    score = (similarity + 1) * 50
    return np.clip(score, 0, 100)


# -----------------------------
# Metric 4 : Composition
# -----------------------------
def composition_score(ref, img):
    ref_moment = cv2.moments(ref)
    img_moment = cv2.moments(img)

    ref_cx = ref_moment["m10"] / (ref_moment["m00"] + 1e-5)
    ref_cy = ref_moment["m01"] / (ref_moment["m00"] + 1e-5)

    img_cx = img_moment["m10"] / (img_moment["m00"] + 1e-5)
    img_cy = img_moment["m01"] / (img_moment["m00"] + 1e-5)

    dist = np.sqrt((ref_cx - img_cx) * 2 + (ref_cy - img_cy) * 2)

    score = 100 - (dist / 128) * 100
    return np.clip(score, 0, 100)


# -----------------------------
# Metric 5 : Detail / Texture
# -----------------------------
def detail_score(ref, img):
    ref_var = cv2.Laplacian(ref, cv2.CV_64F).var()
    img_var = cv2.Laplacian(img, cv2.CV_64F).var()

    diff = abs(ref_var - img_var)
    score = 100 - (diff / max(ref_var, 1)) * 100
    return np.clip(score, 0, 100)


# -----------------------------
# Analyze Reference Importance
# -----------------------------
def reference_weights(ref):

    line_strength = np.mean(cv2.Canny(ref, 100, 200))
    contrast = np.std(ref)
    texture = cv2.Laplacian(ref, cv2.CV_64F).var()

    proportion = 0.30
    line = min(0.30, line_strength / 255)
    value = min(0.25, contrast / 128)
    detail = min(0.20, texture / 500)
    composition = 1.0 - (proportion + line + value + detail)

    weights = {
        "proportion": proportion,
        "line": line,
        "value": value,
        "composition": composition,
        "detail": detail
    }

    total = sum(weights.values())

    for k in weights:
        weights[k] /= total

    return weights


# -----------------------------
# Score Player
# -----------------------------
def score_player(ref, img, weights):

    scores = {
        "proportion": proportion_score(ref, img),
        "line": line_score(ref, img),
        "value": value_score(ref, img),
        "composition": composition_score(ref, img),
        "detail": detail_score(ref, img)
    }

    final = sum(scores[m] * weights[m] for m in scores)

    return final, scores


# -----------------------------
# Judge Endpoint
# -----------------------------
@app.post("/judge")
async def judge(
        reference: UploadFile = File(...),
        playerA: UploadFile = File(...),
        playerB: UploadFile = File(...)
):

    ref_img = load_image(await reference.read())
    a_img = load_image(await playerA.read())
    b_img = load_image(await playerB.read())

    weights = reference_weights(ref_img)

    scoreA, metricsA = score_player(ref_img, a_img, weights)
    scoreB, metricsB = score_player(ref_img, b_img, weights)

    winner = "draw"
    if scoreA > scoreB:
        winner = "playerA"
    elif scoreB > scoreA:
        winner = "playerB"

    return {
        "winner": winner,
        "scoreA": round(scoreA, 2),
        "scoreB": round(scoreB, 2),
        "weights": weights,
        "metricsA": metricsA,
        "metricsB": metricsB
    }
