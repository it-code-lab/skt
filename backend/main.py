# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, io
from PIL import Image
from rdp import rdp
from rembg import remove as rembg_remove
from skimage.segmentation import slic
from skimage.util import img_as_ubyte
from fastapi import Query

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def edges_to_svg_paths(img_bgr: np.ndarray):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    med = float(np.median(gray))
    lo, hi = int(max(0, 0.66*med)), int(min(255, 1.33*med))
    edges = cv2.Canny(gray, lo, hi)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    paths = []
    for c in cnts:
        if len(c) < 16:
            continue
        pts = c[:,0,:].astype(float).tolist()
        simp = rdp(pts, epsilon=2.0)
        if len(simp) < 4:
            continue
        d = f"M {simp[0][0]} {simp[0][1]} " + " ".join([f"L {x} {y}" for x,y in simp[1:]]) + " Z"
        area = abs(cv2.contourArea(np.array(simp, dtype=np.int32)))
        paths.append({"d": d, "area": area, "pts": len(simp)})

    paths.sort(key=lambda p: -p["area"])

    steps = [{"label": f"Stroke {i+1}",
              "paths": [{"d": p["d"]}],
              "est_ms": 600 + 8*p["pts"]} for i,p in enumerate(paths)]

    svg = {
      "viewBox": f"0 0 {img_bgr.shape[1]} {img_bgr.shape[0]}",
      "strokes": [p["d"] for p in paths]
    }
    return svg, steps

# @app.post("/sketch")
# async def sketch(file: UploadFile = File(...)):
#     img = Image.open(io.BytesIO(await file.read())).convert("RGB")
#     img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     svg, steps = edges_to_svg_paths(img_bgr)
#     return JSONResponse({"svg": svg, "steps": steps})

@app.post("/sketch")
async def sketch(
    file: UploadFile = File(...),
    mode: str = Query("auto", enum=["auto","cartoon","photo"]),
    detail: int = Query(5, ge=1, le=10)
):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    svg, steps = build_sketch(img_bgr, mode=mode, detail=detail)
    return JSONResponse({"svg": svg, "steps": steps})

# ---------- utils ----------
def kmeans_quantize(img_bgr, k=8, iters=10):
    import cv2, numpy as np
    Z = img_bgr.reshape((-1,3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, iters, 1.0)
    ret, labels, centers = cv2.kmeans(Z, k, None, criteria, iters, cv2.KMEANS_PP_CENTERS)
    centers = centers.astype(np.uint8)
    q = centers[labels.flatten()].reshape(img_bgr.shape)
    return q

def subject_mask_rembg(img_rgb):
    # returns 8-bit mask 0..255
    out = rembg_remove(img_rgb)  # returns RGBA
    if out.shape[2] == 4:
        alpha = out[:,:,3]
        return alpha
    # fallback
    import cv2, numpy as np
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, m = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    return m

def edges_cartoon(img_bgr, rdp_eps=2.0):
    import cv2, numpy as np
    q = kmeans_quantize(img_bgr, k=7)
    gray = cv2.cvtColor(q, cv2.COLOR_BGR2GRAY)
    # adaptive threshold preserves inner lines in cartoons
    binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 15, 8)
    # seal small gaps; remove dots
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    binv = cv2.morphologyEx(binv, cv2.MORPH_CLOSE, k, iterations=1)
    binv = cv2.morphologyEx(binv, cv2.MORPH_OPEN,  k, iterations=1)
    return binv  # binary edges for contouring

def edges_photo(img_bgr):
    import cv2, numpy as np
    # segment subject, suppress background
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = subject_mask_rembg(rgb)
    mask = cv2.GaussianBlur(mask, (5,5), 0)
    subj = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    gray = cv2.cvtColor(subj, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, 7, 75, 75)  # preserve edges, smooth noise
    med = float(np.median(gray))
    lo, hi = int(max(0, 0.66*med)), int(min(255, 1.33*med))
    edges = cv2.Canny(gray, lo, hi)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, k, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN,  k, iterations=1)
    return edges

def contours_to_steps(binary_img, img_shape, rdp_eps=2.0, min_area_ratio=0.0005,
                      max_paths=350, mode="tree"):
    import cv2, numpy as np
    from rdp import rdp

    h, w = img_shape[:2]
    min_area = max(4, int(min_area_ratio * (h*w)))

    retrieval = cv2.RETR_TREE if mode == "tree" else cv2.RETR_LIST
    cnts, hier = cv2.findContours(binary_img, retrieval, cv2.CHAIN_APPROX_NONE)

    paths = []
    for c in cnts:
        if len(c) < 8:
            continue
        area = abs(cv2.contourArea(c))
        if area < min_area:
            continue
        pts = c[:,0,:].astype(float).tolist()
        simp = rdp(pts, epsilon=rdp_eps)
        if len(simp) < 4:
            continue
        d = f"M {simp[0][0]} {simp[0][1]} " + " ".join([f"L {x} {y}" for x,y in simp[1:]]) + " Z"
        paths.append({"d": d, "area": area, "pts": len(simp)})

    # Keep largest N
    paths.sort(key=lambda p: -p["area"])
    paths = paths[:max_paths]

    steps = [{"label": f"Stroke {i+1}",
              "paths": [{"d": p["d"]}],
              "est_ms": 500 + 7*p["pts"]} for i,p in enumerate(paths)]

    svg = {"viewBox": f"0 0 {w} {h}", "strokes": [p["d"] for p in paths]}
    return svg, steps

def build_sketch(img_bgr, mode="auto", detail=5):
    """
    mode: 'auto' | 'cartoon' | 'photo'
    detail: 1..10 (higher = more detail, more paths)
    """
    import numpy as np
    # map detail to parameters
    rdp_eps = np.interp(detail, [1,10], [4.0, 1.2])   # lower eps -> more points
    min_area_ratio = float(np.interp(detail, [1,10], [0.0020, 0.0002]))
    max_paths = int(np.interp(detail, [1,10], [120, 700]))

    if mode == "auto":
        # quick heuristic: #colors after quantization
        q = kmeans_quantize(img_bgr, k=7)
        # if variance is high, assume photo
        is_photo = (q.std() > 35)
        mode = "photo" if is_photo else "cartoon"

    if mode == "cartoon":
        binary = edges_cartoon(img_bgr)
        return contours_to_steps(binary, img_bgr.shape, rdp_eps, min_area_ratio, max_paths, mode="tree")
    else:  # photo
        edges = edges_photo(img_bgr)
        return contours_to_steps(edges, img_bgr.shape, rdp_eps, min_area_ratio, max_paths, mode="list")