# Use a lightweight PyTorch base image with CUDA or CPU support
FROM pytorch/pytorch:2.1.0-cpu

# Set working directory
WORKDIR /app

# Copy all project files into the container
COPY . .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install flask flask-cors pillow scikit-image torch torchvision

# Ensure the models directory exists and the model file is included
RUN test -f models/ormbg.pth || (echo "Missing model file: models/ormbg.pth" && exit 1)

# Expose port (optional, for local testing)
EXPOSE 5000

# Set the entrypoint to run the Flask app
CMD ["python", "app.py"]
