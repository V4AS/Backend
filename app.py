from flask import Flask, jsonify, request
from flask_cors import CORS
import os
import uuid
import tensorflow as tf
import base64
import sys
import keras_cv
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pickle
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import preprocess_input

app = Flask(__name__)
CORS(app)
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

# setup upload
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# XLA & mixed precision (speed-up processing fel modele generatif check https://www.tensorflow.org/tutorials/generative/generate_images_with_stable_diffusion w ahki aalihom fel rapport) 
tf.config.optimizer.set_jit(True)
tf.keras.mixed_precision.set_global_policy("mixed_float16")

# model generatif
sd_model = keras_cv.models.StableDiffusion(img_width=512, img_height=512, jit_compile=True)
sd_model.diffusion_model.load_weights('./models/kcv_diffusion_model.h5')

# classification model
classification_model = load_model("DLModel.h5")

# Load feature list, file paths, and nearest neighbors model
with open('feature_list.pkl', 'rb') as f:
    feature_list = pickle.load(f)
with open('file_paths.pkl', 'rb') as f:
    file_paths = pickle.load(f)
with open('neighbors_model.pkl', 'rb') as f:
    neighbors = pickle.load(f)

# encode image to base64 (to use fel les 2 endppoints)
def encode_image(image_path):
    with open(image_path, "rb") as fh:
        encoded_string = base64.b64encode(fh.read()).decode('utf-8')
    return encoded_string

# extract features ( to use fel classify model endpoint)
def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return None
    img = cv2.resize(img, (224, 224))
    img = np.array(img)
    expand_img = np.expand_dims(img, axis=0)
    pre_img = preprocess_input(expand_img)
    result = model.predict(pre_img).flatten()
    normalized = result / norm(result)
    return normalized

#modele generatif endpoint
@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.json
    if 'prompt' not in data:
        return jsonify({'error': 'No prompt provided'}), 400

    prompt = data['prompt']
    images = sd_model.text_to_image(prompt, batch_size=1)

    # Save generated image
    filename = str(uuid.uuid4()) + ".png"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    plt.imsave(file_path, images[0])

    response_data = {
        'generatedImageBase64': encode_image(file_path)
    }
    return jsonify(response_data)


#class_model endpoint
@app.route('/classify', methods=['POST'])
def classify_image():
    file = request.files['file']
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    
    query_vector = extract_feature(file_path, classification_model)
    if query_vector is None:
        return jsonify({'error': 'Failed to extract features from image'}), 500

    _, indices = neighbors.kneighbors([query_vector])

    similar_images = [file_paths[idx] for idx in indices[0][:5]]  

    response_data = {
        'similarImagesBase64': [encode_image(img_path) for img_path in similar_images]
    }
    return jsonify(response_data)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)
