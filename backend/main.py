# backend/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2, numpy as np, io
from PIL import Image
from rdp import rdp

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

@app.post("/sketch")
async def sketch(file: UploadFile = File(...)):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    svg, steps = edges_to_svg_paths(img_bgr)
    return JSONResponse({"svg": svg, "steps": steps})
