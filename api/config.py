"""
Configuration settings for the Spam Detection API
"""
import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Model paths
MODEL_DIR = BASE_DIR / "models"
BERT_MODEL_PATH = MODEL_DIR / "bert_model.pth"
BILSTM_MODEL_PATH = MODEL_DIR / "bilstm_model.pth"
CNN_MODEL_PATH = MODEL_DIR / "cnn_model.pth"
VOCAB_PATH = MODEL_DIR / "vocab.pkl"

# API settings
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", 8000))
DEBUG = os.getenv("DEBUG", "false").lower() == "true"

# Model settings
MAX_SEQUENCE_LENGTH = int(os.getenv("MAX_SEQUENCE_LENGTH", 512))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", 32))
DEVICE = os.getenv("DEVICE", "auto")  # auto, cpu, cuda

# Explanation settings
INTEGRATED_GRADIENTS_STEPS = int(os.getenv("IG_STEPS", 20))
GRAD_CAM_CHUNK_SIZE = int(os.getenv("GRAD_CAM_CHUNK_SIZE", 64))

# Email processing settings
MAX_EMAIL_SIZE = int(os.getenv("MAX_EMAIL_SIZE", 10 * 1024 * 1024))  # 10MB
ALLOWED_FILE_EXTENSIONS = [".eml", ".txt", ".msg"]

# Logging settings
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Model loading configuration
MODEL_CONFIG = {
    "bert": {
        "path": BERT_MODEL_PATH,
        "enabled": os.getenv("ENABLE_BERT", "true").lower() == "true",
        "model_name": "bert-base-uncased"
    },
    "bilstm": {
        "path": BILSTM_MODEL_PATH,
        "enabled": os.getenv("ENABLE_BILSTM", "true").lower() == "true",
        "hidden_dim": 128,
        "num_layers": 2
    },
    "cnn": {
        "path": CNN_MODEL_PATH,
        "enabled": os.getenv("ENABLE_CNN", "true").lower() == "true",
        "dropout": 0.5
    }
}

# Ensemble settings
ENSEMBLE_THRESHOLD = float(os.getenv("ENSEMBLE_THRESHOLD", 0.5))
MIN_MODELS_FOR_ENSEMBLE = int(os.getenv("MIN_MODELS_FOR_ENSEMBLE", 1))
