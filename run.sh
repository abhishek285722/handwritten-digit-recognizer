#!/bin/bash
echo "======================================================"
echo "          AI Handwritten Digit Recognizer"
echo "======================================================"
echo ""

# Check python 3.13 or fallback
if command -v py &> /dev/null && py -3.13 --version &> /dev/null; then
    PY_CMD="py -3.13"
elif command -v python3 &> /dev/null; then
    PY_CMD="python3"
else
    PY_CMD="python"
fi

# Fast check: Only run setup if model is missing
if [ ! -f "model.keras" ] && [ ! -f "model.h5" ]; then
    echo "[First-time setup: installing dependencies...]"
    $PY_CMD -m pip install -r requirements.txt
    echo "[Training neural network on MNIST...]"
    $PY_CMD train.py
    if [ $? -ne 0 ]; then
        echo "[ERROR] Training failed."
        exit 1
    fi
fi

echo ""
echo "[Launching web application...]"
$PY_CMD -m streamlit run app.py
