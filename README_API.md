# Spam Detection API

A real-time spam detection API with explainable AI using BERT, BiLSTM, and CNN models. The API accepts raw email messages, makes predictions using all three models, and provides explanations for each model's decision.

## Features

- **Multi-Model Ensemble**: Uses BERT, BiLSTM, and CNN models for robust spam detection
- **Explainable AI**: Provides explanations for each model's predictions:
  - **BERT**: Integrated Gradients for token attribution
  - **BiLSTM**: Attention weights showing token importance
  - **CNN**: Grad-CAM for activation mapping
- **Email Header Integration**: Adds spam detection results as email headers
- **REST API**: FastAPI-based with automatic documentation
- **Docker Support**: Containerized deployment with Docker Compose
- **Comprehensive Testing**: Automated test suite for validation

## Quick Start

### 1. Start the API

```bash
# Using the startup script
./start_api.sh

# Or manually
source venv/bin/activate
export PYTHONPATH=$(pwd)
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Test the API

```bash
# Run comprehensive tests
./test_api.py

# Or test specific functionality
./test_api.py --test spam
```

### 3. Use the API

Visit `http://localhost:8000/docs` for interactive API documentation.

## API Endpoints

### Core Prediction Endpoints

- `POST /predict` - Analyze email content with full predictions and explanations
- `POST /predict/file` - Upload and analyze email files (.eml, .txt, .msg)
- `POST /predict/text` - Analyze plain text for spam

### Information Endpoints

- `GET /health` - Health check and status
- `GET /models/status` - Model loading status
- `GET /explain/{model_name}` - Explanation method information

## Usage Examples

### Python Client

```python
from api.client_example import SpamDetectionClient

client = SpamDetectionClient("http://localhost:8000")

# Analyze email content
email_content = """
From: suspicious@spam.com
Subject: You've won $1,000,000!
...
"""

result = client.predict_email(email_content)
print(f"Spam probability: {result['ensemble_prediction']['score']:.3f}")

# Get explanations
for model, prediction in result['individual_predictions'].items():
    print(f"{model}: {prediction['explanation']['method']}")
```

### cURL Examples

```bash
# Health check
curl http://localhost:8000/health

# Predict spam for text
curl -X POST "http://localhost:8000/predict/text" \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d "text=Get rich quick! Click here now!"

# Upload email file
curl -X POST "http://localhost:8000/predict/file" \
     -F "file=@sample_email.eml" \
     -F "return_format=email"
```

## Model Explanations

### BERT Model
- **Method**: Integrated Gradients
- **Output**: Attribution scores for each token showing contribution to spam prediction
- **Interpretation**: Higher absolute values indicate more important tokens

### BiLSTM Model
- **Method**: Attention Weights
- **Output**: Attention weights showing which tokens the model focuses on
- **Interpretation**: Higher weights indicate tokens the model considers more important

### CNN Model
- **Method**: Grad-CAM (Gradient-weighted Class Activation Mapping)
- **Output**: Activation maps showing important regions in the input
- **Interpretation**: Higher activation values indicate more discriminative regions

## Email Header Integration

When using `return_format="email"`, the API adds the following headers to the original email:

```
X-Spam-Score: 0.8543
X-Spam-Status: SPAM
X-Spam-Models-Used: bert, bilstm, cnn
X-Spam-BERT-Score: 0.9234
X-Spam-BERT-Status: SPAM
X-Spam-BERT-Top-Features: lottery(0.156), money(0.134), click(0.098)
X-Spam-BILSTM-Score: 0.8765
X-Spam-CNN-Score: 0.7632
X-Spam-Agreement: 3/3 models classify as spam
```

## Configuration

Configure the API using environment variables or `api/config.py`:

```bash
# Model settings
export ENABLE_BERT=true
export ENABLE_BILSTM=true
export ENABLE_CNN=true

# API settings
export API_HOST=0.0.0.0
export API_PORT=8000
export DEBUG=false
export LOG_LEVEL=INFO

# Processing settings
export MAX_SEQUENCE_LENGTH=512
export ENSEMBLE_THRESHOLD=0.5
```

## Docker Deployment

### Development

```bash
# Start API only
docker-compose up spam-detection-api

# Start with all services
docker-compose up
```

### Production

```bash
# Start with nginx reverse proxy
docker-compose --profile production up
```

## Model Requirements

Place your trained models in the `models/` directory:

- `bert_model.pth` - Trained BERT model
- `bilstm_model.pth` - Trained BiLSTM model  
- `cnn_model.pth` - Trained CNN model
- `vocab.pkl` - Vocabulary file for BiLSTM/CNN

The API will start even without model files, but will use randomly initialized weights.

## Testing

The test suite validates all API functionality:

```bash
# Run all tests
./test_api.py

# Run specific tests
./test_api.py --test performance
./test_api.py --test spam

# Save detailed results
./test_api.py --output test_results.json
```

## Performance

- **Ensemble prediction**: Combines all model outputs
- **Chunked processing**: Handles large inputs efficiently
- **GPU support**: Automatic CUDA detection and usage
- **Memory optimization**: Gradient checkpointing and chunked processing

## Error Handling

The API provides detailed error messages and handles:

- Invalid email formats
- Missing model files
- Out of memory conditions
- Network timeouts
- File upload errors

## Monitoring

- Health check endpoint for load balancers
- Detailed logging with configurable levels
- Model loading status reporting
- Performance metrics collection

## Security Considerations

- Input validation and sanitization
- File size limits for uploads
- Rate limiting (configure via reverse proxy)
- HTTPS support (configure nginx)

## Development

### Project Structure

```
api/
├── __init__.py
├── main.py              # FastAPI application
├── model_manager.py     # Model loading and inference
├── email_processor.py   # Email parsing and header manipulation
├── config.py           # Configuration settings
└── client_example.py   # Example client usage

models/                 # Trained model files
├── bert_model.pth
├── bilstm_model.pth
├── cnn_model.pth
└── vocab.pkl

test_api.py            # Comprehensive test suite
start_api.sh           # API startup script
```

### Adding New Models

1. Implement model class with required methods
2. Add model configuration to `config.py`
3. Update `ModelManager` to load new model
4. Add explanation method if needed
5. Update tests and documentation

## Troubleshooting

### API Won't Start
- Check Python dependencies: `pip install -r requirements.txt`
- Verify PYTHONPATH: `export PYTHONPATH=$(pwd)`
- Check port availability: `lsof -i :8000`

### Models Not Loading
- Verify model files exist in `models/` directory
- Check file permissions and paths
- Review logs for specific error messages

### Poor Performance
- Enable GPU if available
- Reduce sequence length in config
- Adjust batch size for your hardware
- Use chunked processing for large inputs

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This project is part of a spam detection research implementation.
