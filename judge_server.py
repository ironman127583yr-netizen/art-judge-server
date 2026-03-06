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
# PREPROCESSING
# =========================================================

def preprocess_reference(img):

    blur = cv2.GaussianBlur(img,(5,5),0)
    edges = cv2.Canny(blur,80,160)

    _,thresh = cv2.threshold(edges,40,255,cv2.THRESH_BINARY)

    kernel = np.ones((3,3),np.uint8)
    silhouette = cv2.dilate(thresh,kernel,iterations=1)

    return silhouette


def preprocess_player(img):

    blur = cv2.GaussianBlur(img,(3,3),0)
    edges = cv2.Canny(blur,60,140)

    return edges


# =========================================================
# SKELETONIZATION (gesture AI trick)
# =========================================================

def skeletonize(img):

    _,binary = cv2.threshold(img,40,255,cv2.THRESH_BINARY)

    skeleton = np.zeros(binary.shape,np.uint8)

    element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))

    done = False

    while not done:
        eroded = cv2.erode(binary,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(binary,temp)
        skeleton = cv2.bitwise_or(skeleton,temp)
        binary = eroded.copy()

        if cv2.countNonZero(binary) == 0:
            done = True

    return skeleton


# =========================================================
# FEATURE FUNCTIONS
# =========================================================

def edge_density(img):
    edges = cv2.Canny(img,100,200)
    return float(np.mean(edges))


def value_distribution(img):
    hist = cv2.calcHist([img],[0],None,[256],[0,256])
    return float(np.std(hist))


def composition_balance(img):

    moments = cv2.moments(img)

    if moments["m00"] == 0:
        return 0.0

    cx = moments["m10"]/moments["m00"]
    cy = moments["m01"]/moments["m00"]

    return float(cx + cy)


# =========================================================
# DEPTH
# =========================================================

def depth_measure(img):

    blur = cv2.GaussianBlur(img,(7,7),0)

    grad_x = cv2.Sobel(blur,cv2.CV_64F,1,0,ksize=5)
    grad_y = cv2.Sobel(blur,cv2.CV_64F,0,1,ksize=5)

    magnitude = np.sqrt(grad_x*2 + grad_y*2)

    depth = np.mean(magnitude)

    return float(depth)


# =========================================================
# PROPORTION
# =========================================================

def proportion_measure(img):

    blur = cv2.GaussianBlur(img,(13,13),0)

    _,thresh = cv2.threshold(blur,40,255,cv2.THRESH_BINARY)

    kernel = np.ones((5,5),np.uint8)
    shape = cv2.dilate(thresh,kernel,iterations=2)

    contours = cv2.findContours(
        shape,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )[0]

    if not contours:
        return 0.0

    largest = max(contours,key=cv2.contourArea)

    x,y,w,h = cv2.boundingRect(largest)

    ratio = w/(h+1e-6)

    return float(ratio*100)


# =========================================================
# GESTURE FLOW (skeleton comparison)
# =========================================================

def gesture_similarity(ref,img):

    ref_skel = skeletonize(ref)
    img_skel = skeletonize(img)

    ref_gx = cv2.Sobel(ref_skel,cv2.CV_64F,1,0,ksize=3)
    ref_gy = cv2.Sobel(ref_skel,cv2.CV_64F,0,1,ksize=3)

    img_gx = cv2.Sobel(img_skel,cv2.CV_64F,1,0,ksize=3)
    img_gy = cv2.Sobel(img_skel,cv2.CV_64F,0,1,ksize=3)

    ref_angle = np.arctan2(ref_gy,ref_gx)
    img_angle = np.arctan2(img_gy,img_gx)

    diff = np.abs(ref_angle-img_angle)

    score = 100 - np.mean(diff)*50

    score = max(0,min(100,score))

    return float(score)


# =========================================================
# SCRIBBLE PENALTY
# =========================================================

def stroke_density_penalty(img):

    edges = cv2.Canny(img,100,200)

    density = np.sum(edges>0)/edges.size

    if density > 0.18:
        return 0.6

    if density > 0.12:
        return 0.8

    return 1.0


# =========================================================
# NORMALIZATION (no overshoot reward)
# =========================================================

def normalize(player_feature,ref_feature):

    player_feature=float(player_feature)
    ref_feature=float(ref_feature)

    if ref_feature<=1e-6:
        return 0.0

    ratio = player_feature/ref_feature

    score = 100 - abs(1-ratio)*100

    return float(max(0,min(100,score)))


# =========================================================
# METRIC BALANCING (hierarchy)
# =========================================================

def apply_metric_balance(metrics):

    gesture = metrics["gesture"]
    proportion = metrics["proportion"]

    if gesture < 50:

        metrics["line"] *= 0.7
        metrics["depth"] *= 0.7
        metrics["composition"] *= 0.7

    if proportion < 50:

        metrics["line"] *= 0.8
        metrics["depth"] *= 0.8

    return metrics


# =========================================================
# METRIC COMPUTATION
# =========================================================

def compute_metrics(reference,player):

    ref_line = edge_density(reference)
    ref_value = value_distribution(reference)
    ref_comp = composition_balance(reference)
    ref_depth = depth_measure(reference)
    ref_prop = proportion_measure(reference)

    proportion = normalize(proportion_measure(player),ref_prop)
    line = normalize(edge_density(player),ref_line)
    value = normalize(value_distribution(player),ref_value)
    composition = normalize(composition_balance(player),ref_comp)
    depth = normalize(depth_measure(player),ref_depth)

    gesture = gesture_similarity(reference,player)

    penalty = stroke_density_penalty(player)

    line *= penalty
    depth *= penalty

    metrics = {

        "proportion": proportion,
        "line": line,
        "value": value,
        "gesture": gesture,
        "composition": composition,
        "depth": depth

    }

    metrics = apply_metric_balance(metrics)

    return metrics


# =========================================================
# FINAL SCORE
# =========================================================

def calculate_score(metrics,weights):

    score = 0.0

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

    ref_raw = load_image(await reference.read())
    a_raw = load_image(await playerA.read())
    b_raw = load_image(await playerB.read())

    ref = preprocess_reference(ref_raw)
    a = preprocess_player(a_raw)
    b = preprocess_player(b_raw)

    metricsA = compute_metrics(ref,a)
    metricsB = compute_metrics(ref,b)

    weights = {

        "proportion":0.25,
        "gesture":0.25,
        "line":0.15,
        "value":0.15,
        "depth":0.10,
        "composition":0.10

    }

    scoreA = calculate_score(metricsA,weights)
    scoreB = calculate_score(metricsB,weights)

    winner="draw"

    if scoreA > scoreB:
        winner="playerA"

    elif scoreB > scoreA:
        winner="playerB"

    return {

        "winner":winner,
        "scoreA":round(scoreA,2),
        "scoreB":round(scoreB,2),

        "weights":weights,

        "metricsA":metricsA,
        "metricsB":metricsB

    }
