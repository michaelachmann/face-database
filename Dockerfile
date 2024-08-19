# Dockerfile
FROM python:3.10-slim

# Install gcc and other necessary build tools
RUN apt-get update && apt-get install -y \
    gcc \
    build-essential \
    pkg-config \
    libhdf5-dev

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
CMD ["flask", "run", "--host=0.0.0.0"]
