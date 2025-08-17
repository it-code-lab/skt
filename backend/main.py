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
import cv2, numpy as np
from skimage.morphology import skeletonize
import math

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

# @app.post("/sketch")
# async def sketch(
#     file: UploadFile = File(...),
#     mode: str = Query("auto", enum=["auto","cartoon","photo"]),
#     detail: int = Query(5, ge=1, le=10)
# ):
#     img = Image.open(io.BytesIO(await file.read())).convert("RGB")
#     img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#     svg, steps = build_sketch(img_bgr, mode=mode, detail=detail)
#     return JSONResponse({"svg": svg, "steps": steps})

@app.post("/sketch")
async def sketch(
    file: UploadFile = File(...),
    mode: str = Query("auto", enum=["auto","cartoon","photo"]),
    detail: int = Query(5, ge=1, le=10),
    vector: str = Query("outline", enum=["outline","centerline"])
):
    img = Image.open(io.BytesIO(await file.read())).convert("RGB")
    img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # --- produce a binary edge/ink map as before (your build_sketch) ---
    # Here’s a compact version that gives you a binary for either branch:
    if mode == "auto":
        q = kmeans_quantize(img_bgr, k=7)
        is_photo = (q.std() > 35)
        mode = "photo" if is_photo else "cartoon"

    if mode == "cartoon":
        binary = edges_cartoon(img_bgr)
    else:
        binary = edges_photo(img_bgr)

    # map detail to params
    rdp_eps = float(np.interp(detail, [1,10], [4.0, 1.2]))
    min_area_ratio = float(np.interp(detail, [1,10], [0.0020, 0.0002]))
    max_paths = int(np.interp(detail, [1,10], [120, 700]))

    if vector == "centerline":
        d_list = binary_to_centerline_paths(binary, rdp_eps=rdp_eps, min_len=10)
        svg = {"viewBox": f"0 0 {img_bgr.shape[1]} {img_bgr.shape[0]}", "strokes": d_list}
        steps = [{"label": f"Stroke {i+1}", "paths": [{"d": d}], "est_ms": 600} for i, d in enumerate(d_list)]
        return JSONResponse({"svg": svg, "steps": steps})
    else:
        svg, steps = contours_to_steps(binary, img_bgr.shape, rdp_eps, min_area_ratio, max_paths, retrieval="tree")
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

# def contours_to_steps(binary_img, img_shape, rdp_eps=2.0, min_area_ratio=0.0005,
#                       max_paths=350, mode="tree"):
#     import cv2, numpy as np
#     from rdp import rdp

#     h, w = img_shape[:2]
#     min_area = max(4, int(min_area_ratio * (h*w)))

#     retrieval = cv2.RETR_TREE if mode == "tree" else cv2.RETR_LIST
#     cnts, hier = cv2.findContours(binary_img, retrieval, cv2.CHAIN_APPROX_NONE)

#     paths = []
#     for c in cnts:
#         if len(c) < 8:
#             continue
#         area = abs(cv2.contourArea(c))
#         if area < min_area:
#             continue
#         pts = c[:,0,:].astype(float).tolist()
#         simp = rdp(pts, epsilon=rdp_eps)
#         if len(simp) < 4:
#             continue
#         d = f"M {simp[0][0]} {simp[0][1]} " + " ".join([f"L {x} {y}" for x,y in simp[1:]]) + " Z"
#         paths.append({"d": d, "area": area, "pts": len(simp)})

#     # Keep largest N
#     paths.sort(key=lambda p: -p["area"])
#     paths = paths[:max_paths]

#     steps = [{"label": f"Stroke {i+1}",
#               "paths": [{"d": p["d"]}],
#               "est_ms": 500 + 7*p["pts"]} for i,p in enumerate(paths)]

#     svg = {"viewBox": f"0 0 {w} {h}", "strokes": [p["d"] for p in paths]}
#     return svg, steps

def contours_to_steps(binary_img, img_shape, rdp_eps=2.0, min_area_ratio=0.0005,
                      max_paths=350, retrieval="tree"):
    h, w = img_shape[:2]
    min_area = max(4, int(min_area_ratio * (h*w)))
    mode = cv2.RETR_TREE if retrieval == "tree" else cv2.RETR_LIST
    cnts, _ = cv2.findContours(binary_img, mode, cv2.CHAIN_APPROX_NONE)

    paths = []
    for c in cnts:
        if len(c) < 8: 
            continue
        area = abs(cv2.contourArea(c))
        if area < min_area:
            continue
        if remove_frame_like(c, w, h, area):   # <<< drop page/frame boundary
            continue
        pts = c[:,0,:].astype(float).tolist()
        simp = rdp(pts, epsilon=rdp_eps)
        if len(simp) < 4:
            continue
        d = f"M {simp[0][0]} {simp[0][1]} " + " ".join([f"L {x} {y}" for x,y in simp[1:]]) + " Z"
        paths.append({"d": d, "area": area, "pts": len(simp)})

    paths.sort(key=lambda p: -p["area"])
    paths = paths[:max_paths]
    steps = [{"label": f"Stroke {i+1}", "paths": [{"d": p["d"]}], "est_ms": 500 + 7*p["pts"]}
             for i,p in enumerate(paths)]
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
    
def remove_frame_like(cnt, img_w, img_h, area, pad=3):
    x,y,w,h = cv2.boundingRect(cnt)
    touches_edges = (x <= pad or y <= pad or x + w >= img_w - pad or y + h >= img_h - pad)
    too_big = (w >= 0.95*img_w and h >= 0.95*img_h) or (area >= 0.60 * (img_w*img_h))
    return touches_edges and too_big

# --- replace your binary_to_centerline_paths with this ---


def binary_to_centerline_paths(
    binary,
    rdp_eps=2.0,
    min_len=12,
    bridge_gaps_px=1,        # 0=off, 1-2 good defaults for dashed inputs
    spur_px=6,               # remove hairs shorter than this (pixels)
    join_dist_px=4.0,        # stitch endpoints when this close
    join_angle_deg=25.0      # and roughly collinear (<= this turn)
):
    """
    Convert a binary line image to 1px skeleton, trace long polylines,
    prune spurs, and stitch near-touching endpoints.
    Returns a list of SVG path 'd' strings (centerlines).
    """
    bw = (binary > 0).astype(np.uint8) * 255

    # 1) bridge tiny gaps before thinning (helps dashed segments connect)
    if bridge_gaps_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (2 * bridge_gaps_px + 1, 2 * bridge_gaps_px + 1))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k, iterations=1)

    # 2) skeletonize to 1px
    skel = skeletonize((bw > 0)).astype(np.uint8)

    # utilities
    H, W = skel.shape
    nbrs8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def degree_image(img):
        k = np.ones((3,3), np.uint8)
        # neighbor count = 3x3 sum minus center
        return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_CONSTANT) - img

    def endpoints_and_junctions(img):
        deg = degree_image(img)
        ends = np.argwhere((img == 1) & (deg == 1))
        junc = np.argwhere((img == 1) & (deg >= 3))
        return deg, [tuple(p) for p in ends], [tuple(p) for p in junc]

    # 3) prune short spurs (walk from endpoints until junction/turn; delete if short)
    def prune_spurs(img, maxlen):
        deg, ends, _ = endpoints_and_junctions(img)
        removed = 0
        for ex, ey in ends:
            path = [(ex, ey)]
            cur = (ex, ey)
            prev = None
            for _ in range(maxlen):
                # find the next neighbor (ignore where we came from)
                nxt = None
                for dx, dy in nbrs8:
                    nx, ny = cur[0] + dx, cur[1] + dy
                    if 0 <= nx < H and 0 <= ny < W and img[nx, ny] == 1:
                        if prev is None or (nx, ny) != prev:
                            nxt = (nx, ny)
                            break
                if nxt is None:
                    break
                path.append(nxt)
                prev, cur = cur, nxt
                d = degree_image(img)[cur[0], cur[1]]
                if d != 2:  # hit a junction or another end
                    break
            # remove if the spur is short
            if len(path) <= maxlen and (len(path) < 2 or degree_image(img)[path[-1][0], path[-1][1]] >= 3):
                for px, py in path[:-1]:  # keep the junction pixel
                    img[px, py] = 0
                    removed += 1
        return img, removed

    if spur_px > 0:
        # do a couple of pruning passes
        for _ in range(2):
            skel, _ = prune_spurs(skel, spur_px)

    # 4) trace long polylines (continue through junctions using straightest path)
    visited = np.zeros_like(skel, dtype=np.uint8)
    deg, ends, _ = endpoints_and_junctions(skel)

    def straightest(cur, prev, candidates):
        """Pick neighbor that makes the smallest turn from prev->cur."""
        if prev is None or not candidates:
            return candidates[0] if candidates else None, []
        vx, vy = cur[0] - prev[0], cur[1] - prev[1]
        best = None
        best_ang = 1e9
        rest = []
        for nx, ny in candidates:
            ax, ay = nx - cur[0], ny - cur[1]
            dot = vx*ax + vy*ay
            v1 = math.hypot(vx, vy) or 1.0
            v2 = math.hypot(ax, ay) or 1.0
            ang = math.acos(max(-1.0, min(1.0, dot / (v1*v2))))
            if ang < best_ang:
                if best is not None:
                    rest.append(best)
                best = (nx, ny)
                best_ang = ang
            else:
                rest.append((nx, ny))
        return best, rest

    def neighbors(cur, exclude=None):
        out = []
        for dx, dy in nbrs8:
            nx, ny = cur[0] + dx, cur[1] + dy
            if 0 <= nx < H and 0 <= ny < W and skel[nx, ny] == 1 and not visited[nx, ny]:
                if exclude is None or (nx, ny) != exclude:
                    out.append((nx, ny))
        return out

    seeds = ends[:]  # start from endpoints
    lines = []

    # if there are no endpoints (pure loops), seed from any remaining pixel
    if not seeds:
        seeds = [tuple(p) for p in np.argwhere(skel == 1)]

    while seeds:
        start = seeds.pop()
        if visited[start[0], start[1]]:
            continue
        path = [start]
        visited[start[0], start[1]] = 1
        prev = None
        cur = start

        while True:
            cand = neighbors(cur, exclude=prev)
            if not cand:
                break

            # pick straightest continuation; push other branches as seeds
            nxt, others = straightest(cur, prev, cand)
            for o in others:
                seeds.append(o)

            prev2 = cur
            cur = nxt
            visited[cur[0], cur[1]] = 1
            path.append(cur)
            prev = prev2

            # stop at an endpoint
            if degree_image(skel)[cur[0], cur[1]] == 1:
                break

        if len(path) >= min_len:
            # simplify & convert to (x,y)
            pts_xy = [(p[1], p[0]) for p in path]
            simp = rdp(pts_xy, epsilon=rdp_eps)
            if len(simp) >= 2:
                lines.append(simp)

    # 5) stitch nearly touching endpoints (distance + angle check)
    def endpoints(poly):
        return (poly[0], poly[1]), (poly[-2], poly[-1]) if len(poly) >= 2 else (poly[0], poly[0])

    def direction(a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        l = math.hypot(dx, dy) or 1.0
        return (dx/l, dy/l)

    def angle_between(u, v):
        dot = max(-1.0, min(1.0, u[0]*v[0] + u[1]*v[1]))
        return math.degrees(math.acos(dot))

    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(lines):
            a = lines[i]
            a_s, a_e = a[0], a[-1]
            a_dir_end = direction(a[-2], a[-1]) if len(a) > 1 else (1,0)

            j = i + 1
            merged = False
            while j < len(lines):
                b = lines[j]
                b_s, b_e = b[0], b[-1]
                b_dir_start = direction(b[0], b[1]) if len(b) > 1 else (1,0)
                b_dir_end   = direction(b[-2], b[-1]) if len(b) > 1 else (1,0)

                # try four endpoint pairings (a_end ↔ b_start / b_end, a_start ↔ b_start / b_end)
                pairs = [
                    (a_e, b_s,  a_dir_end,  b_dir_start,  'ae_bs'),
                    (a_e, b_e,  a_dir_end, (-b_dir_end[0], -b_dir_end[1]), 'ae_be'),
                    (a_s, b_s, (-a_dir_end[0], -a_dir_end[1]), b_dir_start, 'as_bs'),
                    (a_s, b_e, (-a_dir_end[0], -a_dir_end[1]),(-b_dir_end[0], -b_dir_end[1]), 'as_be'),
                ]

                did_merge = False
                for P, Q, dP, dQ, mode in pairs:
                    dist = math.hypot(P[0]-Q[0], P[1]-Q[1])
                    ang  = angle_between(dP, dQ)
                    if dist <= join_dist_px and ang <= join_angle_deg:
                        # orient and merge
                        if mode == 'ae_bs':
                            new = a + b
                        elif mode == 'ae_be':
                            new = a + list(reversed(b))
                        elif mode == 'as_bs':
                            new = list(reversed(a)) + b
                        else:  # 'as_be'
                            new = list(reversed(a)) + list(reversed(b))
                        # simplify the join a bit
                        new = rdp(new, epsilon=rdp_eps)
                        lines[i] = new
                        lines.pop(j)
                        did_merge = True
                        merged = True
                        changed = True
                        break
                if not did_merge:
                    j += 1
            if not merged:
                i += 1

    # 6) emit SVG paths
    d_list = [
        f"M {poly[0][0]} {poly[0][1]} " + " ".join(f"L {x} {y}" for (x, y) in poly[1:])
        for poly in lines
        if len(poly) >= 2
    ]

    return d_list


def binary_to_centerline_paths_OLD_DND(binary, rdp_eps=2.0, min_len=12):
    """
    Convert a binary line image to 1px skeleton and trace polylines.
    Returns list of SVG path 'd' strings (centerlines).
    """
    # Ensure single-channel 0/1
    bw = (binary > 0).astype(np.uint8)
    skel = skeletonize(bw).astype(np.uint8)  # 0/1 image

    # Count 8-neighbours for degree
    kernel = np.ones((3,3), np.uint8)
    deg = cv2.filter2D(skel, -1, kernel, borderType=cv2.BORDER_CONSTANT) - skel
    endpoints = np.argwhere((skel == 1) & (deg == 1))
    visited = np.zeros_like(skel, dtype=np.uint8)

    H, W = skel.shape
    nbrs8 = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]

    def trace_from(p):
        path = [tuple(p)]
        visited[p[0], p[1]] = 1
        cur = tuple(p)
        prev = None
        while True:
            nxt = None
            for dx,dy in nbrs8:
                nx, ny = cur[0]+dx, cur[1]+dy
                if 0 <= nx < H and 0 <= ny < W and skel[nx,ny] == 1 and not visited[nx,ny]:
                    if prev is None or (nx,ny) != prev:
                        nxt = (nx,ny)
                        break
            if nxt is None:
                break
            prev = cur
            cur = nxt
            path.append(cur)
            visited[cur[0], cur[1]] = 1
            # stop if we reach a junction (deg>=3) or endpoint (deg==1)
            d = deg[cur[0], cur[1]]
            if d >= 3:
                break
        return path

    # Trace from endpoints first (covers most strokes)
    lines = []
    for p in endpoints:
        p = tuple(p)
        if visited[p[0], p[1]]: 
            continue
        line = trace_from(p)
        if len(line) >= min_len:
            # simplify & build SVG 'd'
            simp = rdp([(xy[1], xy[0]) for xy in line], epsilon=rdp_eps)  # swap to (x,y)
            if len(simp) >= 2:
                d = f"M {simp[0][0]} {simp[0][1]} " + " ".join([f"L {x} {y}" for x,y in simp[1:]])
                lines.append(d)

    # Optional: trace remaining pixels (small loops) as fallback
    remaining = np.argwhere((skel == 1) & (visited == 0))
    for p in remaining:
        p = tuple(p)
        if visited[p[0], p[1]]:
            continue
        line = trace_from(p)
        if len(line) >= min_len:
            simp = rdp([(xy[1], xy[0]) for xy in line], epsilon=rdp_eps)
            if len(simp) >= 2:
                d = f"M {simp[0][0]} {simp[0][1]} " + " ".join([f"L {x} {y}" for x,y in simp[1:]])
                lines.append(d)

    return lines