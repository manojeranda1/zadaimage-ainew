# Use a slim Python 3.10 base image for CPU-only deployment
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY app.py .
COPY models/ormbg.py ./models/
COPY models/ormbg.pth ./models/
COPY static/ ./static/

# Expose port 5000
EXPOSE 5000

# Command to run the Flask application
CMD ["python", "app.py"]