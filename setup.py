import os
import pickle
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.neighbors import NearestNeighbors

input_shape = (224, 224, 3)
num_classes = 5
train_dataset_dir = "Dataset/Animals"
model_save_path = "DLModel.h5"
feature_list_save_path = "feature_list.pkl"
file_paths_save_path = "file_paths.pkl"
neighbors_model_save_path = "neighbors_model.pkl"

# Check DLModel.h5 loaded / not loaded
if not os.path.exists(model_save_path):
    print("Training model...")
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255,
        zoom_range=0.3,
        validation_split=0.2
    )

    train_generator = train_datagen.flow_from_directory(
        train_dataset_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=32,
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        train_dataset_dir,
        target_size=(input_shape[0], input_shape[1]),
        batch_size=32,
        subset='validation'
    )

    base_model = InceptionV3(include_top=False, weights='imagenet', input_shape=input_shape)
    for layer in base_model.layers:
        layer.trainable = False

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.inputs, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint(model_save_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    model.fit(train_generator, validation_data=validation_generator, epochs=10, callbacks=[checkpoint])

# Load the model
model = load_model(model_save_path)

# Feature extraction 
def extract_feature(img_path, model):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = model.predict(img).flatten()
    normalized_features = features / np.linalg.norm(features)
    return normalized_features

# Check feature extraction done / not done
if not os.path.exists(feature_list_save_path) or not os.path.exists(file_paths_save_path):
    print("Extracting features...")
    file_paths = [os.path.join(root, filename) for root, _, filenames in os.walk("Dataset") for filename in filenames]
    feature_list = [extract_feature(file, model) for file in file_paths if extract_feature(file, model) is not None]

    with open(feature_list_save_path, 'wb') as f:
        pickle.dump(feature_list, f)
    with open(file_paths_save_path, 'wb') as f:
        pickle.dump(file_paths, f)

# Train Nearest_Neighbors model
if not os.path.exists(neighbors_model_save_path):
    print("Training Nearest Neighbors model...")
    with open(feature_list_save_path, 'rb') as f:
        feature_list = pickle.load(f)
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(feature_list)
    with open(neighbors_model_save_path, 'wb') as f:
        pickle.dump(neighbors, f)

print("Setup complete.")
