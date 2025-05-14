from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import numpy as np
from PIL import Image, ImageFilter
import rembg
from werkzeug.utils import secure_filename
import logging

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
OUTPUT_SIZE = (2000, 2000)
MAX_PROCESSING_SIZE = 1024

# Initialize rembg session
session = rembg.new_session(model_name="u2netp")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image Resizing Processing
def process_resize(image_data):
    try:
        input_image = Image.open(io.BytesIO(image_data))
        if input_image.mode != 'RGB':
            input_image = input_image.convert('RGB')
        
        output_ratio = max(OUTPUT_SIZE[0] / input_image.width, OUTPUT_SIZE[1] / input_image.height)
        new_size = (int(input_image.width * output_ratio), int(input_image.height * output_ratio))
        
        resized_image = input_image.resize(new_size, Image.Resampling.HAMMING)
        left = (resized_image.width - OUTPUT_SIZE[0]) // 2
        cropped_image = resized_image.crop((left, 0, left + OUTPUT_SIZE[0], OUTPUT_SIZE[1]))
        
        output_bytes = io.BytesIO()
        cropped_image.save(output_bytes, format='PNG', optimize=True)
        output_bytes.seek(0)
        return output_bytes
    except Exception as e:
        logger.error(f"Resize error: {str(e)}")
        raise

def process_bg_removal(image_data):
    try:
        # Open the input image
        input_image = Image.open(io.BytesIO(image_data)).convert('RGBA')

        # Pre-resize image to reduce memory usage
        input_image.thumbnail((MAX_PROCESSING_SIZE, MAX_PROCESSING_SIZE), Image.Resampling.BICUBIC)

        # Perform background removal with alpha matting
        output = rembg.remove(
            input_image,
            session=session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=5
        )
        output = output.convert('RGBA')

        # Optimize alpha channel thresholding
        r, g, b, a = output.split()
        a = a.point(lambda x: 255 if x > 200 else 0)
        output = Image.merge('RGBA', (r, g, b, a))

        # Apply sharpening to enhance edges
        output = output.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))

        # Resize to output size
        output_ratio = min(OUTPUT_SIZE[0] / output.width, OUTPUT_SIZE[1] / output.height)
        new_size = (int(output.width * output_ratio), int(output.height * output_ratio))
        resized_output = output.resize(new_size, Image.Resampling.BICUBIC)

        # Create final image with transparent background
        final_image = Image.new('RGBA', OUTPUT_SIZE, (0, 0, 0, 0))
        position = ((OUTPUT_SIZE[0] - new_size[0]) // 2, (OUTPUT_SIZE[1] - new_size[1]) // 2)
        final_image.paste(resized_output, position, resized_output)

        # Save to bytes
        output_bytes = io.BytesIO()
        final_image.save(output_bytes, format='PNG', optimize=True)
        output_bytes.seek(0)
        return output_bytes
    except Exception as e:
        logger.error(f"Background removal error: {str(e)}")
        raise

@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        processed_image = process_bg_removal(file.read())
        return send_file(
            processed_image,
            mimetype='image/png',
            download_name=f"nobg_{secure_filename(file.filename)}"
        )
    except Exception as e:
        logger.error(f"Background removal endpoint error: {str(e)}")
        return jsonify({"error": f"Background removal failed: {str(e)}"}), 500

@app.route('/api/resize-image', methods=['POST'])
def resize_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        processed_image = process_resize(file.read())
        return send_file(
            processed_image,
            mimetype='image/png',
            download_name=f"resized_{secure_filename(file.filename)}"
        )
    except Exception as e:
        logger.error(f"Resize endpoint error: {str(e)}")
        return jsonify({"error": f"Image resize failed: {str(e)}"}), 500
    
@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "model": "u2netp"})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)