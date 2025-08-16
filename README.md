# Spam Detection with Explainable AI

A comprehensive deep learning project for email spam detection using CNN, BiLSTM, and BERT models with explainability features through LIME and SHAP integration.

## 📋 Project Overview

This project implements and compares three deep learning architectures for spam detection:
- **CNN (Convolutional Neural Network)**: Fast and efficient text classification
- **BiLSTM (Bidirectional LSTM)**: Sequential pattern recognition with attention mechanisms
- **BERT (Bidirectional Encoder Representations from Transformers)**: State-of-the-art transformer-based model

### Key Features
- 🚀 Multiple model architectures for performance comparison
- 🔍 Explainable AI with LIME and SHAP integration
- 📊 Comprehensive evaluation metrics and visualization
- 🐋 Docker support for consistent environments
- 📓 Jupyter notebooks for interactive development
- ⚡ GPU acceleration support (CUDA)

## 🏗️ Project Structure

```
spam-detection/
├── data/                     # Dataset and trained models
├── docs/                     # Research documentation
├── explainability/          # LIME and SHAP explanation modules
├── models/                  # Model implementations (CNN, BiLSTM, BERT)
├── training/                # Training scripts and notebooks
├── metrics/                 # Evaluation metrics and analysis
├── preprocess/              # Data preprocessing utilities
├── utils/                   # Helper functions and constants
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose setup
└── start.sh                # Startup script
```

## 🚀 Getting Started

Choose one of the following methods to run the project:

### Option 1: Google Colab (A100 GPU) - Recommended for BERT

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RationalEar/spam-detection)

1. **Set up the environment in Colab:**
    
    Each notebook already has the necessary setup code. In Google Colab, simply select open notebook and select the GitHub repository:
  https://github.com/RationalEar/spam-detection. 

    You then need to select the runtime, preferably the A100 GPU.

2. **Open the data preparation notebook in Colab:**
   - `preprocess/PrepareDate.ipynb` - Downloads and prepares the dataset

3. **Open any of the training notebooks in Colab:**
   - `training/TrainCNN.ipynb` - CNN model training
   - `training/TrainBiLSTM.ipynb` - BiLSTM model training  
   - `training/TrainBERT.ipynb` - BERT model training

4. **Run the notebooks:**
   - Each notebook contains complete setup and training code
   - Results and models will be saved automatically
   - After training, you can explore the metrics and explainability notebooks:
     - `metrics/*.ipynb`
     - `explainability/*.ipynb`

### Option 2: Local Installation with Python & Jupyter

#### Prerequisites
- Python 3.10+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM (16GB+ recommended for BERT)

#### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RationalEar/spam-detection.git
   cd spam-detection
   ```

2. **Customize file paths:**
   - Open `utils/constants.py` and update the Colab/Windows/Linux paths depending on your environment.

3. **Create virtual environment:**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

4. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Start Jupyter Lab:**
   ```bash
   jupyter lab
   ```

6. **Run the notebooks:**
   - Navigate to `preprocess/` folder
   - Open and run the `PrepareData.ipynb` notebook to download and prepare the dataset
   - Then, open the training notebooks in `training/` folder:
     - `TrainCNN.ipynb` - CNN model training
     - `TrainBiLSTM.ipynb` - BiLSTM model training
     - `TrainBERT.ipynb` - BERT model training
   - For performance evaluation, check the `metrics/` folder and run any of the evaluation notebooks.
   - For explainability, check the `explainability/` folder and run any of the explainability notebooks.


### Option 3: Docker 

#### Prerequisites
- Docker Desktop
- Docker Compose
- NVIDIA Docker (for GPU support)

#### Quick Start

1. **Clone and navigate:**
   ```bash
   git clone https://github.com/RationalEar/spam-detection.git
   cd spam-detection
   ```

2. **Start with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access Jupyter Lab:**
   - Open your browser to `http://localhost:8888`
   - No token required (development setup)

#### Manual Docker Commands

1. **Build the image:**
   ```bash
   docker build -t spam-detection .
   ```

2. **Run with Jupyter Lab:**
   ```bash
   docker run -p 8888:8888 -v ${PWD}:/app spam-detection
   ```

#### GPU Support (NVIDIA Docker)
```bash
# Run with GPU support
docker run --gpus all -p 8888:8888 -v ${PWD}:/app spam-detection
```

## 📋 Requirements

### Core Dependencies
- PyTorch 2.7.1
- Transformers 4.52.4
- Scikit-learn 1.7.0
- LIME 0.2.0.1
- SHAP 0.48.0

### System Requirements
- **Minimum**: 8GB RAM, CPU
- **Local Training**: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM
- **Recommended**: A100 GPU or equivalent for optimal performance


## 📚 Research Paper

For detailed methodology and results, see our research paper in `docs/thesis-short.md`.

## 🎯 Citation

If you use this project in your research, please cite:


## 📞 Support

- 🐛 Issues: [GitHub Issues](https://github.com/RationalEar/spam-detection/issues)

---

