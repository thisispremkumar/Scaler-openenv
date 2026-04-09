FROM python:3.11-slim

WORKDIR /app

# Copy only requirements first (for better layer caching)
COPY server/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Install dependencies in a single layer
COPY . /app
RUN pip install --no-cache-dir .

EXPOSE 7860

# Add healthcheck for faster Spaces startup detection
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:7860/docs', timeout=2)" || exit 1

# Hugging Face Spaces expects the service on port 7860.
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]