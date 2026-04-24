# Base image (Python 3.10 slim is lightweight but contains enough for typical ML)
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Install system dependencies required by OpenCV and Mediapipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy backend requirements
COPY requirements.txt .

# Install Python dependencies
# --no-cache-dir helps keep the docker image size small
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code and model weights
COPY server/ ./server/
COPY kang_mingoo/ ./kang_mingoo/
COPY park_sanghun/ ./park_sanghun/
COPY 신희원/ ./신희원/
COPY 한유승/ ./한유승/

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI using uvicorn
CMD ["python", "-m", "uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000"]
