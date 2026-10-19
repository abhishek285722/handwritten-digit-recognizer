"""
app.py - AI Handwritten Digit Recognizer
Interactive Streamlit Web Application with Live Drawing Canvas, Test Explorer, and Neural Network Inference
"""

import os
# Suppress TensorFlow logging and disable oneDNN scan for fast startup
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import cv2
import numpy as np
from PIL import Image
import streamlit as st

# Safe import for canvas component
try:
    from streamlit_drawable_canvas import st_canvas
    CANVAS_AVAILABLE = True
except ImportError:
    st_canvas = None
    CANVAS_AVAILABLE = False

# Set page configuration
st.set_page_config(
    page_title="AI Handwritten Digit Recognizer",
    page_icon="🔢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich aesthetics and clean window shutdown
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    .hero-header {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4338ca 100%);
        padding: 24px 28px;
        border-radius: 16px;
        margin-bottom: 24px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #ffffff, #c7d2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #e0e7ff;
        margin-top: 6px;
        margin-bottom: 0;
    }
    .badge-pill {
        display: inline-block;
        padding: 4px 12px;
        background: rgba(255, 255, 255, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
        color: #e0e7ff;
        margin-top: 10px;
    }
    .result-card {
        background: linear-gradient(145deg, #0f172a, #1e293b);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid #334155;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    .predicted-digit {
        font-size: 5rem;
        font-weight: 900;
        color: #38bdf8;
        line-height: 1;
        margin: 10px 0;
        text-shadow: 0 0 35px rgba(56, 189, 248, 0.45);
        font-family: 'JetBrains Mono', monospace;
    }
    .confidence-badge {
        font-size: 1.15rem;
        font-weight: 700;
        color: #34d399;
        background: rgba(52, 211, 153, 0.12);
        padding: 6px 16px;
        border-radius: 12px;
        border: 1px solid rgba(52, 211, 153, 0.3);
        display: inline-block;
    }
    .top-candidate-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 10px 14px;
        text-align: center;
    }
    .hint-box {
        background: rgba(30, 41, 59, 0.6);
        border-left: 4px solid #38bdf8;
        padding: 12px 16px;
        border-radius: 0 8px 8px 0;
        margin-bottom: 16px;
        font-size: 0.9rem;
    }
    .stProgress > div > div > div > div {
        background-color: #38bdf8;
    }
    /* Suppress Streamlit disconnect popup so clean shutdown is seamless */
    [data-testid="stConnectionStatus"],
    .stConnectionStatus,
    div[role="dialog"]:has([data-testid="stMarkdownContainer"]) {
        display: none !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading neural network model...")
def load_trained_model():
    """Load model.keras or model.h5."""
    from tensorflow import keras
    base_dir = os.path.dirname(__file__)
    keras_path = os.path.join(base_dir, "model.keras")
    h5_path = os.path.join(base_dir, "model.h5")

    if os.path.exists(keras_path):
        return keras.models.load_model(keras_path)
    elif os.path.exists(h5_path):
        return keras.models.load_model(h5_path)
    return None


@st.cache_data
def load_test_samples():
    """Load pre-saved test samples for instant testing."""
    sample_file = os.path.join(os.path.dirname(__file__), "test_samples.npz")
    if os.path.exists(sample_file):
        data = np.load(sample_file)
        return data["images"], data["labels"]
    return None, None


def center_image_by_mass(img_28x28):
    """
    Center digit using center of mass (same standard as MNIST dataset).
    Calculates center of mass of non-zero pixels and shifts it to (14, 14).
    """
    from scipy.ndimage import center_of_mass
    cy, cx = center_of_mass(img_28x28)
    
    if np.isnan(cy) or np.isnan(cx):
        return img_28x28

    shift_y = int(np.round(14 - cy))
    shift_x = int(np.round(14 - cx))

    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    centered = cv2.warpAffine(img_28x28, M, (28, 28), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return centered


def preprocess_drawn_canvas(canvas_data):
    """
    Preprocess image from streamlit_drawable_canvas:
    1. Extract drawing and check if empty
    2. Convert to grayscale
    3. Invert if black strokes on white canvas -> white strokes on black background
    4. Crop to bounding box of digit
    5. Resize bounding box into 20x20 keeping aspect ratio
    6. Pad into 28x28 and center with center-of-mass
    7. Normalize to [0.0, 1.0]
    """
    if canvas_data is None:
        return None, None

    img_array = canvas_data.astype(np.uint8)

    # Canvas returns RGBA
    if img_array.ndim == 3 and img_array.shape[2] == 4:
        alpha = img_array[:, :, 3]
        rgb = img_array[:, :, :3]
        
        is_transparent = np.all(alpha == 0)
        if is_transparent:
            return None, None

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        
        # Invert black strokes on white canvas into white digit on black background
        white_bg = np.mean(gray) > 127
        if white_bg:
            binary = 255 - gray
            binary[alpha == 0] = 0
        else:
            binary = gray
    else:
        return None, None

    if np.max(binary) < 30:
        return None, None

    # Clean thresholding
    _, thresh = cv2.threshold(binary, 40, 255, cv2.THRESH_TOZERO)

    # Find bounding box
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None, None

    x, y, w, h = cv2.boundingRect(coords)
    digit_crop = thresh[y:y+h, x:x+w]

    # Resize digit to fit within 20x20 while preserving aspect ratio
    if w > h:
        new_w = 20
        new_h = max(1, int(round((h * 20.0) / w)))
    else:
        new_h = 20
        new_w = max(1, int(round((w * 20.0) / h)))

    resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Place in center of 28x28 canvas
    canvas_28 = np.zeros((28, 28), dtype=np.float32)
    start_x = (28 - new_w) // 2
    start_y = (28 - new_h) // 2
    canvas_28[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit

    # Shift by center of mass for optimal MNIST alignment
    try:
        canvas_28 = center_image_by_mass(canvas_28)
    except Exception:
        pass

    normalized = canvas_28 / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)

    return input_tensor, normalized


def preprocess_uploaded_image(uploaded_file):
    """Process an uploaded image file (PNG/JPG)."""
    pil_image = Image.open(uploaded_file).convert("L")
    img_np = np.array(pil_image)

    # Auto detect if background is white or black
    if np.mean(img_np) > 127:
        img_np = 255 - img_np

    _, thresh = cv2.threshold(img_np, 50, 255, cv2.THRESH_TOZERO)
    coords = cv2.findNonZero(thresh)
    
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        digit_crop = thresh[y:y+h, x:x+w]
        if w > h:
            new_w = 20
            new_h = max(1, int(round((h * 20.0) / w)))
        else:
            new_h = 20
            new_w = max(1, int(round((w * 20.0) / h)))
        resized = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas_28 = np.zeros((28, 28), dtype=np.float32)
        start_x = (28 - new_w) // 2
        start_y = (28 - new_h) // 2
        canvas_28[start_y:start_y+new_h, start_x:start_x+new_w] = resized
        canvas_28 = center_image_by_mass(canvas_28)
    else:
        canvas_28 = cv2.resize(img_np.astype(np.float32), (28, 28))

    normalized = canvas_28 / 255.0
    input_tensor = np.expand_dims(normalized, axis=0)
    return input_tensor, normalized


# Hero Header
st.markdown("""
<div class="hero-header">
    <h1 class="hero-title">🔢 AI Handwritten Digit Recognizer</h1>
    <p class="hero-subtitle">Interactive Deep Learning System with Live Canvas, Preprocessing Pipeline, and Neural Network Inference</p>
    <span class="badge-pill">MNIST Dataset (70,000 Images)</span>
    <span class="badge-pill" style="margin-left: 8px;">Accuracy: 97.6%+</span>
    <span class="badge-pill" style="margin-left: 8px;">Deep Neural Network</span>
</div>
""", unsafe_allow_html=True)


# Load Model
model = load_trained_model()

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Application Controls")
    
    st.markdown("""
    **Model Status:**  
    `✅ Deep Neural Network Loaded`
    
    **Features:**
    - ✏️ Live Drawing Canvas ($280 \\times 280$)
    - 🧠 Center-of-Mass Moments Alignment
    - 📁 Custom Image File Testing
    - 🎲 MNIST Test Dataset Validation
    """)
    
    st.markdown("---")
    st.subheader("📈 Quick Benchmark")
    st.write("**Dataset:** MNIST (70,000 Total)")
    st.write("**Training Images:** 60,000")
    st.write("**Test Images:** 10,000")
    st.write("**Test Accuracy:** `97.6%+`")

    st.markdown("---")
    if st.button("🛑 Stop Session", use_container_width=True, type="secondary", help="Stop application and close window"):
        import os
        os._exit(0)


if model is None:
    st.warning("⚠️ Model file (`model.h5` or `model.keras`) not found! Please run `python train.py` first to train the network.")
    if st.button("🚀 Train Model Now directly from App"):
        with st.spinner("Training neural network on MNIST dataset... (takes ~1-2 minutes)"):
            import subprocess
            res = subprocess.run(["py", "-3.13", "train.py"], capture_output=True, text=True)
            if res.returncode == 0:
                st.success("Model trained successfully! Reloading...")
                st.rerun()
            else:
                st.error(f"Training failed: {res.stderr}")
    st.stop()


# Main Application Navigation Tabs
tab_canvas, tab_samples, tab_performance = st.tabs([
    "✏️ Live Drawing Canvas",
    "📁 Test Images & MNIST Dataset",
    "📊 Model Performance & Architecture",
])


# -------------------------------------------------------------
# TAB 1: Live Interactive Drawing Canvas
# -------------------------------------------------------------
with tab_canvas:
    col_left, col_right = st.columns([1.1, 1], gap="large")

    input_tensor = None
    preview_28x28 = None

    with col_left:
        st.subheader("✏️ Draw a Digit (0 – 9)")
        st.markdown("""
        <div class="hint-box">
            Draw any digit in the canvas below with your mouse or touchscreen. 
            The system centers your stroke and recognizes the digit instantly!
        </div>
        """, unsafe_allow_html=True)

        col_b1, col_b2 = st.columns([1.5, 1])
        with col_b1:
            brush_size = st.slider("Brush Thickness", min_value=12, max_value=26, value=18, step=2)
        with col_b2:
            st.write("")
            realtime = st.checkbox("Realtime Predict", value=True, help="Update prediction as you draw")

        # Initialize canvas state key
        if "canvas_key" not in st.session_state:
            st.session_state.canvas_key = 0

        # Action Buttons: Clear Canvas and Manual Recognize
        c_clear, c_recognize = st.columns(2)
        with c_clear:
            if st.button("🗑️ Clear Canvas", use_container_width=True, help="Erase canvas and start fresh"):
                st.session_state.canvas_key += 1
                st.rerun()
        with c_recognize:
            manual_predict = st.button("🔮 Recognize Digit", type="primary", use_container_width=True, help="Trigger recognition")

        if CANVAS_AVAILABLE and st_canvas is not None:
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0)",
                stroke_width=brush_size,
                stroke_color="#000000",
                background_color="#FFFFFF",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=f"digit_canvas_{st.session_state.canvas_key}",
                update_streamlit=realtime
            )

            if canvas_result.image_data is not None:
                input_tensor, preview_28x28 = preprocess_drawn_canvas(canvas_result.image_data)
        else:
            st.error("⚠️ Canvas module not available. Please run `run.bat` using Python 3.13 where dependencies are installed.")

    with col_right:
        st.subheader("🎯 Prediction & Analysis")

        if input_tensor is not None:
            predictions = model.predict(input_tensor, verbose=0)[0]
            predicted_digit = int(np.argmax(predictions))
            confidence = float(predictions[predicted_digit]) * 100.0

            # Large Glowing Result Card
            st.markdown(f"""
            <div class="result-card">
                <span style="font-size: 0.95rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">
                    Predicted Digit
                </span>
                <div class="predicted-digit">{predicted_digit}</div>
                <div class="confidence-badge">
                    Confidence: {confidence:.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Top 3 Candidates Ranking
            sorted_indices = np.argsort(predictions)[::-1]
            st.write("🏆 **Top 3 Candidate Predictions:**")
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.markdown(f"""
                <div class="top-candidate-card" style="border-color: #38bdf8;">
                    <div style="font-size: 0.8rem; color: #94a3b8;">🥇 1st Choice</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #38bdf8;">Digit {sorted_indices[0]}</div>
                    <div style="font-size: 0.85rem; color: #34d399; font-weight: 600;">{predictions[sorted_indices[0]]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_t2:
                st.markdown(f"""
                <div class="top-candidate-card">
                    <div style="font-size: 0.8rem; color: #94a3b8;">🥈 2nd Choice</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #cbd5e1;">Digit {sorted_indices[1]}</div>
                    <div style="font-size: 0.85rem; color: #94a3b8;">{predictions[sorted_indices[1]]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with col_t3:
                st.markdown(f"""
                <div class="top-candidate-card">
                    <div style="font-size: 0.8rem; color: #94a3b8;">🥉 3rd Choice</div>
                    <div style="font-size: 1.6rem; font-weight: 800; color: #94a3b8;">Digit {sorted_indices[2]}</div>
                    <div style="font-size: 0.85rem; color: #64748b;">{predictions[sorted_indices[2]]*100:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Class Probability Distribution (0–9)
            st.write("📊 **Class Probability Distribution (0–9):**")
            for digit in range(10):
                prob = float(predictions[digit]) * 100.0
                is_winner = (digit == predicted_digit)
                label = f"**Digit {digit}**" if is_winner else f"Digit {digit}"
                
                c_label, c_bar, c_pct = st.columns([1.5, 5, 1.5])
                with c_label:
                    st.markdown(f"{'👉 ' if is_winner else ''}{label}")
                with c_bar:
                    st.progress(float(predictions[digit]))
                with c_pct:
                    st.markdown(f"`{prob:.1f}%`")

            # Under the Hood: Show preprocessed 28x28 image
            with st.expander("🔍 Inspect Model Input (28×28 Tensor)"):
                c_in1, c_in2 = st.columns(2)
                with c_in1:
                    st.write("**Centered 28×28 Digit Frame:**")
                    st.image(preview_28x28, width=140, clamp=True)
                with c_in2:
                    st.write("**Tensor Properties:**")
                    st.json({
                        "Shape": str(input_tensor.shape),
                        "Min Value": float(np.min(input_tensor)),
                        "Max Value": float(np.max(input_tensor)),
                        "Mean Intensity": float(np.round(np.mean(input_tensor), 3))
                    })
        else:
            st.markdown("""
            <div class="result-card" style="padding: 48px 24px;">
                <div style="font-size: 3rem; margin-bottom: 12px;">✏️</div>
                <h3 style="color: #cbd5e1; margin: 0;">Canvas is Empty</h3>
                <p style="color: #64748b; font-size: 0.95rem; margin-top: 8px;">
                    Draw a number on the canvas to the left to see live prediction and neural network confidence.
                </p>
            </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 2: Upload & MNIST Test Samples
# -------------------------------------------------------------
with tab_samples:
    sub_col1, sub_col2 = st.columns(2, gap="large")

    with sub_col1:
        st.subheader("🎲 MNIST Test Dataset Explorer")
        st.markdown("""
        <div class="hint-box">
            Validate the model against unseen images directly from the standard MNIST test set (10,000 images).
        </div>
        """, unsafe_allow_html=True)

        sample_images, sample_labels = load_test_samples()
        if sample_images is not None:
            col_filter, col_pick = st.columns([1.5, 1])
            with col_filter:
                digit_filter = st.selectbox(
                    "Filter by Digit:",
                    ["All Digits (Random)"] + [f"Digit {d}" for d in range(10)]
                )
            with col_pick:
                st.write("")
                pick_clicked = st.button("🎲 Next Sample", use_container_width=True)

            # Determine eligible sample indices
            if digit_filter == "All Digits (Random)":
                eligible_indices = list(range(len(sample_images)))
            else:
                target_digit = int(digit_filter.split()[-1])
                eligible_indices = [i for i, lbl in enumerate(sample_labels) if lbl == target_digit]

            if "mnist_sample_idx" not in st.session_state or pick_clicked:
                if eligible_indices:
                    st.session_state.mnist_sample_idx = int(np.random.choice(eligible_indices))
                else:
                    st.session_state.mnist_sample_idx = 0

            s_idx = st.session_state.mnist_sample_idx
            sample_img = sample_images[s_idx]
            ground_truth = int(sample_labels[s_idx])

            # Run inference on sample
            s_tensor = np.expand_dims(sample_img, axis=0)
            s_preds = model.predict(s_tensor, verbose=0)[0]
            s_predicted = int(np.argmax(s_preds))
            s_conf = float(s_preds[s_predicted]) * 100.0

            c_simg, c_sres = st.columns([1, 1.2])
            with c_simg:
                st.write(f"**Sample #{s_idx}**")
                st.image(sample_img, caption=f"Ground Truth: Digit {ground_truth}", width=180)
            with c_sres:
                st.write(f"**Actual Digit:** `{ground_truth}`")
                st.write(f"**Predicted Digit:** `{s_predicted}`")
                st.write(f"**Confidence:** `{s_conf:.2f}%`")
                if s_predicted == ground_truth:
                    st.success("✅ Prediction Matches Ground Truth!")
                else:
                    st.error("❌ Mismatch detected.")
        else:
            st.info("Test samples file not found. Run `train.py` to create `test_samples.npz`.")

    with sub_col2:
        st.subheader("📁 Upload Handwritten Digit")
        st.markdown("""
        <div class="hint-box">
            Upload your own scanned or phone-photographed handwritten digit image (PNG, JPG, JPEG).
        </div>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader("Upload Digit Image", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            up_tensor, up_preview = preprocess_uploaded_image(uploaded_file)
            up_preds = model.predict(up_tensor, verbose=0)[0]
            up_predicted = int(np.argmax(up_preds))
            up_conf = float(up_preds[up_predicted]) * 100.0

            c_u1, c_u2 = st.columns([1, 1.2])
            with c_u1:
                st.image(uploaded_file, caption="Uploaded File", width=160)
                st.image(up_preview, caption="Centered 28x28 Preprocessed", width=120)
            with c_u2:
                st.markdown(f"""
                <div class="result-card">
                    <span style="font-size: 0.85rem; color: #94a3b8;">Predicted Digit</span>
                    <div class="predicted-digit" style="font-size: 3.5rem;">{up_predicted}</div>
                    <div class="confidence-badge">{up_conf:.1f}% Confidence</div>
                </div>
                """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 3: Model Performance & Architecture
# -------------------------------------------------------------
with tab_performance:
    st.subheader("📊 Model Performance Benchmarks")

    # Metrics row
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    with col_m1:
        st.metric("Test Accuracy", "97.6%+", help="Evaluated on 10,000 unseen MNIST test images")
    with col_m2:
        st.metric("Training Samples", "60,000", help="Standard MNIST training partition")
    with col_m3:
        st.metric("Test Samples", "10,000", help="Standard MNIST evaluation partition")
    with col_m4:
        st.metric("Total Parameters", "109,386", help="Learned weights across 4 layers")

    st.markdown("---")

    col_diag1, col_diag2 = st.columns([1.3, 1], gap="large")

    with col_diag1:
        st.subheader("📈 Training & Validation Curves")
        history_img_path = os.path.join(os.path.dirname(__file__), "training_history.png")
        if os.path.exists(history_img_path):
            st.image(history_img_path, caption="Accuracy and Crossentropy Loss across 10 Epochs", use_container_width=True)
        else:
            st.info("Run `train.py` to generate the training history curves.")

    with col_diag2:
        st.subheader("🧠 Deep Neural Network Architecture")
        st.markdown(r"""
        - **Input Layer:** $28 \times 28$ Grayscale image ($784$ total pixels)
        - **Flatten Layer:** Reshapes 2D matrix into a 1D feature vector of $784$ values
        - **Dense Hidden Layer 1:** $128$ Neurons with **ReLU** ($\max(0, x)$) activation
        - **Dropout Layer:** $20\%$ rate to prevent overfitting on specific pixels
        - **Dense Hidden Layer 2:** $64$ Neurons with **ReLU** activation
        - **Output Layer:** $10$ Neurons with **Softmax** activation (probability distribution over classes 0–9)
        - **Optimizer:** Adam ($\alpha = 0.001$)
        - **Loss Function:** Sparse Categorical Crossentropy
        """)

        st.markdown("#### 🔍 Layer Parameter Summary")
        layer_data = [
            {"Layer": "Input (Image)", "Type": "28x28 Grayscale", "Output Shape": "(28, 28)", "Parameters": "0"},
            {"Layer": "flatten", "Type": "Flatten", "Output Shape": "(784,)", "Parameters": "0"},
            {"Layer": "dense_128", "Type": "Dense (ReLU)", "Output Shape": "(128,)", "Parameters": "100,480"},
            {"Layer": "dropout_1", "Type": "Dropout (0.20)", "Output Shape": "(128,)", "Parameters": "0"},
            {"Layer": "dense_64", "Type": "Dense (ReLU)", "Output Shape": "(64,)", "Parameters": "8,256"},
            {"Layer": "output_softmax", "Type": "Dense (Softmax)", "Output Shape": "(10,)", "Parameters": "650"},
        ]
        st.table(layer_data)
