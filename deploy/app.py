import os
import argparse

from PIL import Image
from flask import Flask, render_template, request, jsonify, send_file

import torch
import tempfile
from inference import InferenceProcess

import uuid

app = Flask(__name__)

MODEL_LIST = {
    "candy": "../weights/candy.onnx",
    "mosaic": "../weights/mosaic.onnx",
    "rain-princess": "../weights/rain-princess.onnx",
    "udnie": "../weights/udnie.onnx",
}

# Define command-line arguments
parser = argparse.ArgumentParser(description="Deployment Arguments")
parser.add_argument("--port", type=int, default=5000, help="Port number to run the server on")
parser.add_argument(
    "--model",
    type=str,
    default="mosaic", help="Model name 'candy', 'mosaic', 'rain-princess', 'udnie'")
args = parser.parse_args()


def load_model(model_name: str):
    assert model_name in MODEL_LIST, f"Model `{model_name}` is not in Model List: {MODEL_LIST.keys()}"

    inference_instance = InferenceProcess(MODEL_LIST[model_name])

    return inference_instance


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    # Read the image file
    image = request.files['image']
    # Load the model
    inference_instance = load_model(args.model)

    filename = image.name
    image = Image.open(image).convert('RGB')

    # Save the output image to a temporary file
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, f"{str(uuid.uuid4())}.jpg")

    # Perform inference
    with torch.no_grad():
        output = inference_instance(image)

    # Convert the output tensor to an image and save
    save_image(image_path, output[0])

    # Return the path to the output image (temporary file path)
    return jsonify({'output_image_path': image_path})


@app.route('/get_image/<image_name>')
def get_image(image_name):
    temp_dir = tempfile.gettempdir()
    image_path = os.path.join(temp_dir, image_name)
    return send_file(image_path, mimetype='image/png')


def save_image(image_path, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(image_path)


if __name__ == '__main__':
    app.run(port=args.port, debug=True)
