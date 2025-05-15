# Use a valid PyTorch CPU image with Python 3.10
FROM pytorch/pytorch:2.1.0-cpu-py3.10

# Set working directory
WORKDIR /app

# Copy everything into the container
COPY . .

# Install required packages
RUN pip install --upgrade pip && \
    pip install flask flask-cors pillow scikit-image

# Ensure model file exists
RUN test -f models/ormbg.pth || (echo "Missing model file: models/ormbg.pth" && exit 1)

# Expose port for Flask
EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
