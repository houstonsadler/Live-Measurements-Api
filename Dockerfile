# Use a small Python base image
FROM python:3.10-slim

# Keep installs quiet/clean
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System libs OpenCV/MediaPipe need + git for torch.hub (MiDaS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# App directory
WORKDIR /app

# Install Python deps first (better Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . /app

# Render will inject $PORT at runtime; bind Gunicorn to it
ENV PORT=8000

# Start the production server (Flask app object is `app` in app.py)
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:${PORT}", "--workers", "2", "--threads", "4", "--timeout", "180"]
