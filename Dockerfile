FROM python:3.11-slim

WORKDIR /app

# Minimal native libs needed by opencv-python-headless
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app.py /app/app.py
COPY templates /app/templates
COPY manifest.json /app/manifest.json
COPY service-worker.js /app/service-worker.js
COPY icon-192.png /app/icon-192.png
COPY icon-512.png /app/icon-512.png

ENV PORT=5000
EXPOSE 5000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} app:app"]
