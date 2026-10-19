"""
test_model.py - Automated verification tests for Handwritten Digit Recognizer
"""

import os
import sys
import numpy as np

def test_environment():
    """Verify that all core libraries can be imported."""
    print("Testing core imports...")
    import tensorflow as tf
    from tensorflow import keras
    import streamlit
    import streamlit_drawable_canvas
    import cv2
    import PIL
    import matplotlib
    import scipy
    print("All core dependencies imported successfully!")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"Keras Version:      {keras.__version__}")
    return True


def test_model_inference():
    """Verify that model can be loaded and infers on 28x28 input."""
    base_dir = os.path.dirname(__file__)
    h5_path = os.path.join(base_dir, "model.h5")
    keras_path = os.path.join(base_dir, "model.keras")

    if not os.path.exists(h5_path) and not os.path.exists(keras_path):
        print("Model file does not exist yet. Please run train.py first.")
        return False

    from tensorflow import keras
    target_path = keras_path if os.path.exists(keras_path) else h5_path
    print(f"Loading model from: {target_path}")
    model = keras.models.load_model(target_path)
    
    # Create synthetic test tensor (1, 28, 28)
    dummy_input = np.zeros((1, 28, 28), dtype=np.float32)
    # Draw simple vertical line (like digit 1)
    dummy_input[0, 5:23, 13:15] = 1.0

    predictions = model.predict(dummy_input, verbose=0)
    assert predictions.shape == (1, 10), f"Expected shape (1, 10), got {predictions.shape}"
    predicted_digit = int(np.argmax(predictions[0]))
    confidence = float(predictions[0][predicted_digit])

    print(f"Inference test passed! Predicted digit: {predicted_digit} (Confidence: {confidence * 100:.2f}%)")
    return True


if __name__ == "__main__":
    env_ok = test_environment()
    if env_ok:
        test_model_inference()
