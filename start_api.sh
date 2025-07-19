#!/bin/bash

# Spam Detection API Startup Script

echo "Starting Spam Detection API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Set PYTHONPATH to include the project root
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create models directory if it doesn't exist
mkdir -p models

# Check if model files exist
MODEL_DIR="models"
echo "Checking for model files in $MODEL_DIR..."

if [ ! -f "$MODEL_DIR/bert_model.pth" ]; then
    echo "Warning: BERT model not found at $MODEL_DIR/bert_model.pth"
fi

if [ ! -f "$MODEL_DIR/bilstm_model.pth" ]; then
    echo "Warning: BiLSTM model not found at $MODEL_DIR/bilstm_model.pth"
fi

if [ ! -f "$MODEL_DIR/cnn_model.pth" ]; then
    echo "Warning: CNN model not found at $MODEL_DIR/cnn_model.pth"
fi

if [ ! -f "$MODEL_DIR/vocab.pkl" ]; then
    echo "Warning: Vocabulary file not found at $MODEL_DIR/vocab.pkl"
fi

echo "Note: The API will start even without model files, but predictions will not work."
echo "Make sure to train and save your models to the models/ directory."

# Start the API server
echo "Starting FastAPI server on http://localhost:8000"
echo "API documentation will be available at http://localhost:8000/docs"

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
