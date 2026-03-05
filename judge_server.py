from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Art Judge Server Running"}


def load_image(file_bytes):
    image = Image.open(io.BytesIO(file_bytes)).convert("L")
    image = image.resize((128, 128))
    return np.array(image)


def calculate_similarity(img1, img2):
    diff = np.abs(img1.astype("float") - img2.astype("float"))
    score = 100 - (np.mean(diff) / 255) * 100
    return max(0, min(100, score))


def edge_similarity(img1, img2):
    edge1 = cv2.Canny(img1, 100, 200)
    edge2 = cv2.Canny(img2, 100, 200)

    diff = np.abs(edge1.astype("float") - edge2.astype("float"))
    score = 100 - (np.mean(diff) / 255) * 100
    return max(0, min(100, score))


@app.post("/judge")
async def judge(
    reference: UploadFile = File(...),
    playerA: UploadFile = File(...),
    playerB: UploadFile = File(...)
):

    ref_img = load_image(await reference.read())
    a_img = load_image(await playerA.read())
    b_img = load_image(await playerB.read())

    shapeA = edge_similarity(ref_img, a_img)
    shapeB = edge_similarity(ref_img, b_img)

    valueA = calculate_similarity(ref_img, a_img)
    valueB = calculate_similarity(ref_img, b_img)

    scoreA = (shapeA * 0.6) + (valueA * 0.4)
    scoreB = (shapeB * 0.6) + (valueB * 0.4)

    winner = "draw"
    if scoreA > scoreB:
        winner = "playerA"
    elif scoreB > scoreA:
        winner = "playerB"

    return {
        "scoreA": round(scoreA, 2),
        "scoreB": round(scoreB, 2),
        "winner": winner,
        "metricsA": {
            "shape": round(shapeA, 2),
            "value": round(valueA, 2)
        },
        "metricsB": {
            "shape": round(shapeB, 2),
            "value": round(valueB, 2)
        }
    }
