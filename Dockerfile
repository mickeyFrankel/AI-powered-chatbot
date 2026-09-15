FROM python:3.11-slim

WORKDIR /app

# System deps some of the requirements need to build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt python-multipart

# Backend code + the pre-seeded vector DB (bundled into the image -
# no external DB service to provision for this demo)
COPY backend/ ./backend/
COPY databases/vector/ ./databases/vector/
COPY config/ ./config/

# Pre-built frontend (vite build output) - served by FastAPI itself
COPY frontend/dist/ ./frontend/dist/

# HF Spaces Docker SDK expects the app on port 7860
ENV PORT=7860
EXPOSE 7860

WORKDIR /app/backend/core
CMD ["python3", "api.py"]
