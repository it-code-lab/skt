=========================================
=========================================
Backend (FastAPI + OpenCV) — Setup & Run
=========================================
=========================================
cd sketcher
mkdir -p backend && cd backend


# Windows:
python -m venv venv

venv\Scripts\activate

python -m pip install --upgrade pip
pip install fastapi "uvicorn[standard]" python-multipart pillow opencv-python shapely rdp


Run back end:
uvicorn main:app --reload --port 8000

Quick test (optional):

Swagger: http://127.0.0.1:8000/docs


curl -X POST -F "file=@/path/to/any-image.jpg" http://127.0.0.1:8000/sketch


=========================================
=========================================
Frontend (React + Vite + TypeScript) — Setup & Run
=========================================
=========================================

cd frontend
npm create vite@latest . -- --template react-ts
npm install
npm install framer-motion
npm run dev

Open the URL it prints (usually http://localhost:5173).
Upload any image — you should see the stroke-by-stroke drawing.

===============================
=====================================
Getting image for the input: (Flat-color cartoon)
=====================================
=====================================

[OBJECT/CHARACTER], [view: side/3-quarter/front], clean vector cartoon, 
bold black outlines ~3–5px, flat colors (4–6), high contrast, 
separate parts clearly visible ([list key parts]), centered, 
plain white background, no gradients, no texture, no drop shadow, no text, no watermark, 
high resolution, [ASPECT e.g., 16:9 or 9:16], 2048 px long side

UI Input for such image ==>Cartoon Line art, Detail - 5, Centerline strokes


===============================
=====================================
Getting image for the input: (Pure line art)
=====================================
=====================================

Black-ink line art of [OBJECT/CHARACTER], [view], single continuous strokes, 
uniform line weight, no fill, no shading, no cross-hatching, 
clean outlines with closed gaps, centered, white background, 
no texture, no text, no watermark, high resolution, [ASPECT], 2048 px long side


UI Input for such image ==>Auto, Detail - 5, Centerline strokes