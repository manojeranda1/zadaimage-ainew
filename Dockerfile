FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
WORKDIR /app
COPY . /app
RUN pip install flask flask-cors pillow scikit-image werkzeug
COPY ormbg.pth /app/ormbg.pth  # Copy to root folder
CMD ["python", "app.py"]