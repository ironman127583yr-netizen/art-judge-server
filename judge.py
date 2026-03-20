import numpy as np
import cv2
from PIL import Image
import io
import requests

# =========================
# IMAGE LOAD
# =========================

def load_image_from_url(url):
    response = requests.get(url)
    response.raise_for_status()
    img = Image.open(io.BytesIO(response.content)).convert("L")
    img = img.resize((256, 256))
    return np.array(img)

# =========================
# PROCESSING
# =========================

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

# =========================
# MAIN JUDGE
# =========================

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
