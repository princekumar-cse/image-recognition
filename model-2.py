"""
Model loading and prediction logic for the Image Recognition project.
Uses MobileNetV2 pretrained on ImageNet (transfer learning — no training
needed, we reuse a network Google already trained on 1.2M images).
"""

import time
import numpy as np
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.utils import img_to_array


def load_model():
    """Loads MobileNetV2 with weights trained on ImageNet."""
    return MobileNetV2(weights="imagenet")


def predict_image(model, pil_image):
    """
    Takes a PIL image (already resized to 224x224) and the loaded model,
    returns (top_5_predictions, elapsed_ms).

    top_5_predictions is a list of (label, probability) tuples.
    """
    image_array = img_to_array(pil_image)
    image_batch = np.expand_dims(image_array, axis=0)
    processed_image = preprocess_input(image_batch)

    start_time = time.time()
    predictions = model.predict(processed_image, verbose=0)
    elapsed_ms = (time.time() - start_time) * 1000

    decoded = decode_predictions(predictions, top=5)[0]
    top_5 = [(label, float(probability)) for (_, label, probability) in decoded]

    return top_5, elapsed_ms
