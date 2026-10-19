"""
train.py - Handwritten Digit Recognizer Training Pipeline
Dataset: MNIST (60,000 train, 10,000 test images of handwritten digits 0-9)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Reduce TensorFlow verbose logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def load_and_preprocess_data():
    """Load MNIST dataset and normalize pixel intensities to [0, 1]."""
    print("=" * 60)
    print("Step 1 & 2: Loading MNIST dataset from Keras...")
    print("=" * 60)
    
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    print(f"Training data shape: {x_train.shape}, labels: {y_train.shape}")
    print(f"Test data shape:     {x_test.shape}, labels: {y_test.shape}")
    print(f"Pixel range before:  [{x_train.min()}, {x_train.max()}]")

    # Step 3: Normalize pixel values (0-255 -> 0.0-1.0)
    print("\nStep 3: Normalizing pixel values to [0.0, 1.0]...")
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    print(f"Pixel range after:   [{x_train.min():.1f}, {x_train.max():.1f}]")

    # Save a small set of representative test samples for instant offline UI testing
    sample_data_path = os.path.join(os.path.dirname(__file__), "test_samples.npz")
    np.savez_compressed(
        sample_data_path,
        images=x_test[:100],
        labels=y_test[:100]
    )
    print(f"Saved 100 sample test images to '{sample_data_path}' for web UI preview.")

    return (x_train, y_train), (x_test, y_test)


def build_neural_network():
    """
    Step 4: Construct Neural Network architecture
    Input (28x28) -> Flatten (784) -> Dense 128 (ReLU) -> Dropout(0.2) -> Dense 64 (ReLU) -> Dense 10 (Softmax)
    """
    print("\n" + "=" * 60)
    print("Step 4: Building Neural Network Architecture...")
    print("=" * 60)

    model = keras.Sequential([
        layers.Input(shape=(28, 28)),
        layers.Flatten(),
        layers.Dense(128, activation="relu", name="dense_128"),
        layers.Dropout(0.2, name="dropout_1"),
        layers.Dense(64, activation="relu", name="dense_64"),
        layers.Dense(10, activation="softmax", name="output_softmax")
    ])

    model.summary()
    return model


def compile_and_train(model, x_train, y_train, epochs=10, batch_size=32):
    """Step 5 & 6: Compile and train the neural network."""
    print("\n" + "=" * 60)
    print("Step 5: Compiling model with Adam and Sparse Categorical Crossentropy...")
    print("=" * 60)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\n" + "=" * 60)
    print(f"Step 6: Training for {epochs} epochs (batch size: {batch_size}, validation split: 10%)...")
    print("=" * 60)

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.1,
        verbose=1
    )
    return history


def evaluate_model(model, x_test, y_test):
    """Step 7: Evaluate performance on 10,000 unseen test images."""
    print("\n" + "=" * 60)
    print("Step 7: Evaluating on 10,000 unseen test images...")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)
    print("\n" + "-" * 40)
    print(f"Final Test Loss:     {test_loss:.4f}")
    print(f"Final Test Accuracy: {test_acc * 100:.2f}%")
    print("-" * 40)
    return test_loss, test_acc


def save_trained_model(model):
    """Step 8: Save trained model to disk."""
    base_dir = os.path.dirname(__file__)
    h5_path = os.path.join(base_dir, "model.h5")
    keras_path = os.path.join(base_dir, "model.keras")

    print("\n" + "=" * 60)
    print("Step 8: Saving model to disk...")
    print("=" * 60)

    # Save H5 format
    model.save(h5_path)
    print(f"Saved model to: {h5_path}")

    # Save native .keras format for Keras 3 compatibility
    try:
        model.save(keras_path)
        print(f"Saved native Keras format to: {keras_path}")
    except Exception as e:
        print(f"Note on native keras save: {e}")


def plot_and_save_history(history):
    """Step 9: Generate and save training loss and accuracy plots."""
    print("\n" + "=" * 60)
    print("Step 9: Plotting training history curves...")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    output_png = os.path.join(base_dir, "training_history.png")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs_range = range(1, len(history.history["accuracy"]) + 1)

    # Accuracy Plot
    ax1.plot(epochs_range, history.history["accuracy"], "o-", label="Training Accuracy", color="#3b82f6", linewidth=2)
    ax1.plot(epochs_range, history.history["val_accuracy"], "s--", label="Validation Accuracy", color="#10b981", linewidth=2)
    ax1.set_title("Model Accuracy Across Epochs", fontsize=14, fontweight="bold", pad=12)
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Accuracy", fontsize=12)
    ax1.set_xticks(epochs_range)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="lower right", fontsize=11)

    # Loss Plot
    ax2.plot(epochs_range, history.history["loss"], "o-", label="Training Loss", color="#ef4444", linewidth=2)
    ax2.plot(epochs_range, history.history["val_loss"], "s--", label="Validation Loss", color="#f59e0b", linewidth=2)
    ax2.set_title("Model Loss Across Epochs", fontsize=14, fontweight="bold", pad=12)
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Crossentropy Loss", fontsize=12)
    ax2.set_xticks(epochs_range)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=11)

    plt.suptitle("Handwritten Digit Recognizer (MNIST) - Training Performance", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Training curve chart successfully saved to: {output_png}")


def main():
    print(f"Using TensorFlow Version: {tf.__version__}")
    print(f"Using Keras Version:      {keras.__version__}")
    
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    model = build_neural_network()
    history = compile_and_train(model, x_train, y_train, epochs=10, batch_size=32)
    evaluate_model(model, x_test, y_test)
    save_trained_model(model)
    plot_and_save_history(history)

    print("\nTraining and evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
