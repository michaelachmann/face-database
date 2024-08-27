# Use the argument ENVIRONMENT to switch between GPU and CPU
ARG ENVIRONMENT=CPU

# Stage 1: Base image for GPU-enabled systems
FROM nvidia/cuda:11.8-cudnn8-runtime-ubuntu22.04 AS gpu

# Install gcc and other necessary build tools for GPU environment
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Base image for CPU-only systems
FROM python:3.10-slim AS cpu

# Install gcc and other necessary build tools for CPU environment
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Choose the final stage based on the ENVIRONMENT variable
FROM ${ENVIRONMENT,,} AS final

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the Flask port
EXPOSE 8080

# Run the Flask app
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]
