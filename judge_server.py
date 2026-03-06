from fastapi import FastAPI, UploadFile, File
import numpy as np
import cv2
from PIL import Image
import io

app = FastAPI()

# =========================================================
# ROOT
# =========================================================

@app.get("/")
def home():
    return {"status": "Art Judge Server Running"}


# =========================================================
# IMAGE LOADING
# =========================================================

def load_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("L")
    img = img.resize((256,256))
    return np.array(img)


# =========================================================
# UNIFIED PREPROCESSING PIPELINE
# =========================================================

def build_structural_maps(img):

    blur = cv2.GaussianBlur(img,(5,5),0)

    edges = cv2.Canny(blur,70,150)

    _,binary = cv2.threshold(blur,40,255,cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    silhouette = cv2.morphologyEx(binary,cv2.MORPH_CLOSE,kernel)

    gradient_x = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=3)
    gradient_y = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=3)

    magnitude = np.hypot(gradient_x, gradient_y)

    return {
        "edges":edges,
        "silhouette":silhouette,
        "gradient_mag":magnitude,
        "blur":blur
    }


# =========================================================
# SKELETONIZATION
# =========================================================

def skeletonize(binary):

    skeleton = np.zeros(binary.shape,np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    img = binary.copy()

    done = False

    while not done:

        eroded = cv2.erode(img,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(img,temp)

        skeleton = cv2.bitwise_or(skeleton,temp)

        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            done = True

    return skeleton


# =========================================================
# METRICS
# =========================================================

def silhouette_similarity(ref,player):

    ref_bin = ref["silhouette"] > 0
    player_bin = player["silhouette"] > 0

    intersection = np.logical_and(ref_bin,player_bin)
    union = np.logical_or(ref_bin,player_bin)

    score = intersection.sum()/(union.sum()+1e-6)

    return float(score*100)


def gesture_similarity(ref,player):

    ref_skel = skeletonize(ref["silhouette"])
    player_skel = skeletonize(player["silhouette"])

    ref_gx = cv2.Sobel(ref_skel,cv2.CV_64F,1,0,ksize=3)
    ref_gy = cv2.Sobel(ref_skel,cv2.CV_64F,0,1,ksize=3)

    player_gx = cv2.Sobel(player_skel,cv2.CV_64F,1,0,ksize=3)
    player_gy = cv2.Sobel(player_skel,cv2.CV_64F,0,1,ksize=3)

    ref_angle = np.arctan2(ref_gy,ref_gx)
    player_angle = np.arctan2(player_gy,player_gx)

    diff = np.abs(ref_angle-player_angle)

    score = 100 - np.mean(diff)*50

    return float(max(0,min(100,score)))


def proportion_measure(struct):

    contours,_ = cv2.findContours(
        struct["silhouette"],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return 0.0

    largest = max(contours,key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(largest)

    return float(w/(h+1e-6))


def depth_measure(struct):

    contours,_ = cv2.findContours(
        struct["silhouette"],
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) <= 1:
        return 0.0

    areas = [cv2.contourArea(c) for c in contours]

    return float(np.std(areas))


def line_quality(struct):

    edges = struct["edges"]
    silhouette = struct["silhouette"]

    skeleton = skeletonize(silhouette)

    edge_pixels = np.sum(edges > 0)
    skeleton_pixels = np.sum(skeleton > 0)

    if edge_pixels == 0:
        return 0

    efficiency = skeleton_pixels / (edge_pixels + 1e-6)

    score = efficiency * 100

    return float(max(0,min(100,score)))


def composition_balance(struct):

    m = cv2.moments(struct["silhouette"])

    if m["m00"] == 0:
        return 0

    cx = m["m10"]/m["m00"]
    cy = m["m01"]/m["m00"]

    center_dist = np.sqrt((cx-128)*2 + (cy-128)*2)

    score = 100 - center_dist/2

    return float(max(0,min(100,score)))


# =========================================================
# NORMALIZATION
# =========================================================

def normalize(player,ref):

    if ref <= 1e-6:
        return 0.0

    ratio = player/ref

    score = 100 - abs(1-ratio)*100

    return float(max(0,min(100,score)))


def apply_structure_hierarchy(metrics):

    silhouette = metrics["silhouette"]
    gesture = metrics["gesture"]
    proportion = metrics["proportion"]

    structure_score = (silhouette + gesture + proportion) / 3

    if structure_score < 60:
        metrics["line"] *= 0.6
        metrics["depth"] *= 0.6
        metrics["composition"] *= 0.6

    return metrics


# =========================================================
# METRIC COMPUTATION
# =========================================================

def compute_metrics(ref_struct,player_struct):

    silhouette = silhouette_similarity(ref_struct,player_struct)

    gesture = gesture_similarity(ref_struct,player_struct)

    # prevent scribble gesture cheating
    gesture = min(gesture, silhouette + 10)

    proportion = normalize(
        proportion_measure(player_struct),
        proportion_measure(ref_struct)
    )

    line = line_quality(player_struct)

    depth = normalize(
        depth_measure(player_struct),
        depth_measure(ref_struct)
    )

    composition = composition_balance(player_struct)

    metrics = {
        "silhouette": silhouette,
        "gesture": gesture,
        "proportion": proportion,
        "line": line,
        "depth": depth,
        "composition": composition
    }

    metrics = apply_structure_hierarchy(metrics)

    return metrics


# =========================================================
# FINAL SCORE
# =========================================================

def calculate_score(metrics,weights):

    score = 0

    for key in weights:
        score += metrics[key]*weights[key]

    return float(score)


# =========================================================
# JUDGE ENDPOINT
# =========================================================

@app.post("/judge")
async def judge(
    reference: UploadFile = File(...),
    playerA: UploadFile = File(...),
    playerB: UploadFile = File(...)
):

    ref_img = load_image(await reference.read())
    a_img = load_image(await playerA.read())
    b_img = load_image(await playerB.read())

    ref_struct = build_structural_maps(ref_img)
    a_struct = build_structural_maps(a_img)
    b_struct = build_structural_maps(b_img)

    metricsA = compute_metrics(ref_struct,a_struct)
    metricsB = compute_metrics(ref_struct,b_struct)

    weights = {
        "silhouette":0.30,
        "gesture":0.25,
        "proportion":0.15,
        "line":0.15,
        "composition":0.10,
        "depth":0.05
    }

    scoreA = calculate_score(metricsA,weights)
    scoreB = calculate_score(metricsB,weights)

    winner = "draw"

    if scoreA > scoreB:
        winner = "playerA"
    elif scoreB > scoreA:
        winner = "playerB"

    return {
        "winner":winner,
        "scoreA":round(scoreA,2),
        "scoreB":round(scoreB,2),
        "weights":weights,
        "metricsA":metricsA,
        "metricsB":metricsB
    }
