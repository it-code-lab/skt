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