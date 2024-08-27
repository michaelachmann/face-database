# ARG for environment selection
ARG ENVIRONMENT=cpu

# Stage 1: Base image for GPU-enabled systems
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04 AS gpu

# Stage 2: Base image for CPU-only systems
FROM python:3.10-slim AS cpu

# Conditional final stage based on ENVIRONMENT argument
FROM ${ENVIRONMENT} AS final

# Install gcc and other necessary build tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    gcc \
    build-essential \
    pkg-config \
    libhdf5-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

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