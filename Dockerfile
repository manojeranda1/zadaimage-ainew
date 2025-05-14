# Use a lightweight Python 3.10 base image from Debian Bookworm
FROM python:3.10-slim-bookworm

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies required for Pillow, scikit-image etc.
# Using --no-install-recommends and cleaning apt cache keeps the image smaller
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    libtiff-dev \
    libwebp-dev && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install Python dependencies
# This step is cached by Docker if requirements.txt doesn't change
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
# This includes app.py, the models directory, and the static directory
COPY . .

# Ensure the static directory exists (optional, your code does this too)
RUN mkdir -p static

# Expose the port the Flask app runs on
EXPOSE 5000

# Command to run the Flask application using the Python interpreter
# Use the exec form for proper signal handling
CMD ["python", "app.py"]