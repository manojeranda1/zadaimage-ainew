from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import io
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from werkzeug.utils import secure_filename
import logging
import urllib.request

# Create models folder and download model files if not already present
os.makedirs("models", exist_ok=True)

ormbg_py_path = os.path.join("models", "ormbg.py")
ormbg_pth_path = os.path.join("models", "ormbg.pth")

if not os.path.exists(ormbg_py_path):
    urllib.request.urlretrieve("https://saas.zada.lk/models/ormbg.py", ormbg_py_path)

if not os.path.exists(ormbg_pth_path):
    urllib.request.urlretrieve("https://saas.zada.lk/models/ormbg.pth", ormbg_pth_path)

# Import ORMBG model class dynamically after download
from models.ormbg import ORMBG

app = Flask(__name__, static_url_path='', static_folder='static')
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
OUTPUT_SIZE = (2000, 2000)        # Final output canvas size
MODEL_INPUT_SIZE = [1024, 1024]   # Input size expected by ORMBG model
MODEL_PATH = ormbg_pth_path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    logger.info(f"Loading model from {MODEL_PATH}, size: {os.path.getsize(MODEL_PATH)} bytes")

    model = ORMBG()
    # Load state_dict directly (weights_only=False param removed because torch.load doesn't accept it)
    state_dict = torch.load(MODEL_PATH, map_location=device)
    
    if not isinstance(state_dict, dict):
        raise TypeError("The loaded model file is not a state_dict. Please check the model file.")
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(im: np.ndarray, model_input_size: list) -> torch.Tensor:
    # Ensure 3 channels
    if len(im.shape) == 2:
        im = np.stack([im]*3, axis=-1)
    im_tensor = torch.tensor(im, dtype=torch.float32).permute(2, 0, 1)
    im_tensor = F.interpolate(
        im_tensor.unsqueeze(0), size=model_input_size, mode="bilinear"
    ).type(torch.uint8)
    image = im_tensor.float() / 255.0
    return image

def postprocess_image(result: torch.Tensor, im_size: list) -> np.ndarray:
    result = F.interpolate(result.unsqueeze(0), size=im_size, mode="bilinear").squeeze(0)
    ma, mi = result.max(), result.min()
    normalized = (result - mi) / (ma - mi + 1e-8)  # add epsilon to avoid div by zero
    im_array = (normalized * 255).permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    if im_array.shape[2] == 1:
        im_array = np.squeeze(im_array, axis=2)
    return im_array

def remove_background_with_ormbg(image_stream):
    try:
        orig_image = Image.open(image_stream).convert("RGB")
        orig_np = np.array(orig_image)
        orig_size = orig_np.shape[:2]  # (height, width)

        image = preprocess_image(orig_np, MODEL_INPUT_SIZE).to(device)

        with torch.no_grad():
            result = model(image)

        result_mask = postprocess_image(result[0][0], orig_size)

        alpha_mask = Image.fromarray(result_mask).convert("L")
        rgba_image = orig_image.convert("RGBA")
        rgba_image.putalpha(alpha_mask)

        return rgba_image
    except Exception as e:
        logger.error(f"Background removal error: {e}")
        raise

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
        logger.error(f"Resize error: {e}")
        raise

@app.route('/api/remove-background', methods=['POST'])
def remove_background():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid file"}), 400

    try:
        result_img = remove_background_with_ormbg(file.stream)

        # Resize and center on transparent background
        result_ratio = min(OUTPUT_SIZE[0] / result_img.width, OUTPUT_SIZE[1] / result_img.height)
        new_size = (int(result_img.width * result_ratio), int(result_img.height * result_ratio))
        resized = result_img.resize(new_size, Image.Resampling.LANCZOS)

        final_image = Image.new('RGBA', OUTPUT_SIZE, (0, 0, 0, 0))
        position = ((OUTPUT_SIZE[0] - new_size[0]) // 2, (OUTPUT_SIZE[1] - new_size[1]) // 2)
        final_image.paste(resized, position, resized)

        img_bytes = io.BytesIO()
        final_image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        return send_file(
            img_bytes,
            mimetype='image/png',
            download_name=f"nobg_{secure_filename(file.filename)}"
        )
    except Exception as e:
        logger.error(f"Background removal endpoint error: {e}")
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
        logger.error(f"Resize endpoint error: {e}")
        return jsonify({"error": f"Image resize failed: {str(e)}"}), 500

@app.route('/health')
def health_check():
    return jsonify({"status": "ok", "model": "ormbg"})

@app.route('/')
def index():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    os.makedirs('static', exist_ok=True)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
