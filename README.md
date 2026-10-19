# 🔢 Handwritten Digit Recognizer (MNIST)
**Deep Learning Handwritten Digit Classifier**

An interactive Deep Learning web application built with **TensorFlow / Keras** and **Streamlit** that recognizes handwritten digits (0–9) drawn with a mouse or uploaded from files. The underlying neural network is trained on the classic **MNIST dataset** (70,000 images) and achieves **97.6%+ test accuracy**.

---

## 🌟 Key Features

1. **✏️ Live Interactive Drawing Canvas (`streamlit-drawable-canvas`):**
   - Draw digits in real time on a $280 \times 280$ responsive canvas.
   - Dynamic brush thickness control (12px – 26px) with instant undo and clear.
   - Real-time prediction mode for instant feedback as you draw.

2. **🧠 Smart MNIST Preprocessing Pipeline:**
   - Automatic stroke color inversion (converting white-canvas/black-strokes into MNIST standard white-strokes/black-background).
   - Non-zero bounding box cropping to eliminate empty canvas borders.
   - Aspect-ratio preserved scaling into a $20 \times 20$ box.
   - **Center of Mass (Moments) Alignment:** Centers the digit mass at $(14, 14)$ inside the $28 \times 28$ matrix, exactly mirroring standard MNIST preprocessing.

3. **📊 Probability Distribution (Softmax Confidence):**
   - Displays real-time class probability breakdown for all 10 digits ($0$ through $9$).
   - Highlights the winning digit with a large glowing prediction card.

4. **📁 Dual Testing Modes:**
   - **Canvas Mode:** Draw digits freely.
   - **Upload Mode:** Upload your own PNG/JPG handwritten digit.
   - **Random MNIST Sample Mode:** Pick random samples from the 10,000 unseen test set to test the model against known ground truth.

5. **📈 Visual Training Diagnostics:**
   - Generates and embeds `training_history.png` showing Accuracy and Crossentropy Loss progression over all 10 epochs.

---

## 🏗️ Neural Network Architecture

```text
       Input Image (28 × 28 Grayscale)
                    │
                    ▼
     ┌─────────────────────────────┐
     │ Flatten Layer (784 features)│
     └──────────────┬──────────────┘
                    ▼
     ┌─────────────────────────────┐
     │ Dense Layer (128 Neurons)   │─── ReLU Activation
     └──────────────┬──────────────┘
                    ▼
     ┌─────────────────────────────┐
     │ Dropout Layer (Rate: 0.20)  │─── Regularization to prevent overfitting
     └──────────────┬──────────────┘
                    ▼
     ┌─────────────────────────────┐
     │ Dense Layer (64 Neurons)    │─── ReLU Activation
     └──────────────┬──────────────┘
                    ▼
     ┌─────────────────────────────┐
     │ Dense Layer (10 Neurons)    │─── Softmax Activation
     └──────────────┬──────────────┘
                    ▼
        Predicted Digit Class (0 - 9)
        + Confidence % (0.0% - 100.0%)
```

### Layer Parameters Breakdown
| Layer | Type | Output Shape | Param # | Activation |
|---|---|---|---|---|
| `flatten` | Flatten | `(None, 784)` | 0 | None |
| `dense_128` | Dense | `(None, 128)` | 100,480 | ReLU |
| `dropout_1` | Dropout (0.2) | `(None, 128)` | 0 | None |
| `dense_64` | Dense | `(None, 64)` | 8,256 | ReLU |
| `output_softmax`| Dense | `(None, 10)` | 650 | Softmax |
| **Total** | | | **109,386 parameters** | |

---

## 📊 Dataset & Training Specifications

- **Dataset:** MNIST (`keras.datasets.mnist.load_data()`)
- **Total Samples:** 70,000 images ($28 \times 28$, single channel grayscale)
  - **Training Set:** 60,000 images (with 10% validation split = 54k train, 6k val)
  - **Test Set:** 10,000 unseen images
- **Normalization:** $\frac{\text{pixel}}{255.0} \in [0.0, 1.0]$
- **Optimizer:** Adam ($\alpha = 0.001$)
- **Loss Function:** Sparse Categorical Crossentropy:
  $$\mathcal{L} = -\sum_{i=1}^{N} \log(p_{y_i})$$
- **Metrics:** Categorical Accuracy
- **Epochs:** 10
- **Batch Size:** 32

### Performance Benchmarks
- **Training Accuracy:** $\approx 98.9\%$
- **Validation Accuracy:** $\approx 97.8\%$
- **Test Accuracy (10,000 unseen images):** $\ge 97.6\%$

---

## 📁 Project Structure

```text
handwritten digit recognizer/
│
├── train.py                  # Training pipeline (loads MNIST, trains NN, plots history, saves model)
├── app.py                    # Interactive Streamlit application with live drawing canvas
├── test_model.py             # Automated unit verification tests
├── requirements.txt          # Project dependencies (TensorFlow, Keras, Streamlit, etc.)
├── run.bat                   # 1-click Windows launcher (targets Python 3.13)
├── run.sh                    # 1-click Linux/Mac launcher
├── model.h5                  # Trained neural network weights & architecture
├── model.keras               # Modern native Keras 3 model format
├── training_history.png      # Epoch-by-epoch Accuracy & Loss curves
├── test_samples.npz          # 100 pre-saved test images for instant offline UI testing
├── README.md                 # Complete project documentation
└── .streamlit/
    └── config.toml           # Streamlit performance configuration
```

---

## 🚀 Quick Start Guide

### Option 1: 1-Click Launcher (Recommended for Windows)
Simply double-click:
```cmd
run.bat
```
This script automatically:
1. Detects Python 3.13.
2. Checks and installs required dependencies.
3. Automatically trains the model if `model.h5` does not exist yet.
4. Opens the Streamlit app in your default browser.

### Option 2: Manual Terminal Execution
```bash
# 1. Install dependencies
py -3.13 -m pip install -r requirements.txt

# 2. Train the neural network
py -3.13 train.py

# 3. Launch the web application
py -3.13 -m streamlit run app.py
```

---

## 🎥 Video Demonstration Tips
1. **Show the Live Canvas:** Draw 2–3 different digits (e.g. `3`, `7`, `8`) and highlight how the confidence bar dynamically jumps to the correct class.
2. **Explain the Preprocessing:** Point out how real mouse strokes are resized, centered by center-of-mass, and normalized just like MNIST.
3. **Show the Architecture:** Explain the dense layers, dropout for regularization, and softmax probabilities.
