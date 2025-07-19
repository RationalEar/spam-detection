from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import os
import tempfile
import logging
from datetime import datetime

from api.model_manager import ModelManager
from api.email_processor import EmailProcessor
from api.config import (
    MODEL_CONFIG, API_HOST, API_PORT, DEBUG, MAX_EMAIL_SIZE,
    ALLOWED_FILE_EXTENSIONS, LOG_LEVEL, LOG_FORMAT, ENSEMBLE_THRESHOLD
)

# Configure logging
logging.basicConfig(level=getattr(logging, LOG_LEVEL), format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Spam Detection API",
    description="Real-time spam detection with explainable AI using BERT, BiLSTM, and CNN models",
    version="1.0.0",
    debug=DEBUG
)

# Global variables for models and processors
model_manager: Optional[ModelManager] = None
email_processor: EmailProcessor = EmailProcessor()

# Pydantic models for request/response
class EmailRequest(BaseModel):
    email_content: str = Field(..., description="Raw email content as string")
    return_format: str = Field(default="json", description="Response format: 'json' or 'email'")


class PredictionResponse(BaseModel):
    timestamp: str
    ensemble_prediction: Dict[str, Any]
    individual_predictions: Dict[str, Dict[str, Any]]
    email_headers_added: bool = False
    modified_email: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global model_manager
    
    logger.info("Starting Spam Detection API...")

    # Build model paths from configuration
    model_paths = {}
    for model_name, config in MODEL_CONFIG.items():
        if config.get('enabled', True):
            model_paths[model_name] = str(config['path'])

    vocab_path = str(MODEL_CONFIG.get('vocab_path', 'models/vocab.pkl'))

    try:
        model_manager = ModelManager(model_paths, vocab_path)
        logger.info(f"API startup complete. Loaded {len(model_manager.models)} models")

        if model_manager.failed_models:
            logger.warning(f"Some models failed to load: {model_manager.failed_models}")

    except Exception as e:
        logger.error(f"Error during startup: {e}")
        # Initialize with empty model paths to allow API to start
        model_manager = ModelManager({}, vocab_path)


@app.get("/")
async def root():
    """Root endpoint with API information"""
    global model_manager

    models_loaded = 0
    failed_models = []
    if model_manager:
        models_loaded = len(model_manager.models)
        failed_models = model_manager.failed_models

    return {
        "message": "Spam Detection API",
        "version": "1.0.0",
        "description": "Real-time spam detection with explainable AI",
        "status": {
            "models_loaded": models_loaded,
            "failed_models": failed_models,
            "device": str(model_manager.device) if model_manager else "unknown"
        },
        "endpoints": {
            "POST /predict": "Analyze email content for spam",
            "POST /predict/file": "Upload and analyze email file",
            "POST /predict/text": "Analyze plain text for spam",
            "GET /models/status": "Check model loading status",
            "GET /health": "Health check",
            "GET /explain/{model_name}": "Get explanation method info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global model_manager
    
    models_loaded = 0
    if model_manager:
        models_loaded = len(model_manager.models)
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": models_loaded
    }


@app.get("/models/status")
async def models_status():
    """Check status of loaded models"""
    global model_manager
    
    if not model_manager:
        return {"status": "No model manager initialized"}
    
    status = {
        "total_models": len(model_manager.models),
        "loaded_models": list(model_manager.models.keys()),
        "device": str(model_manager.device) if hasattr(model_manager, 'device') else "unknown"
    }
    
    return status


@app.post("/predict", response_model=PredictionResponse)
async def predict_spam(request: EmailRequest):
    """
    Analyze email content for spam detection with explanations
    """
    global model_manager, email_processor
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    if not model_manager.models:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    try:
        # Parse email content
        parsed_email = email_processor.parse_raw_email(request.email_content)
        combined_text = parsed_email['combined_text']
        
        # Get predictions from all models
        predictions = model_manager.predict_all(combined_text)
        
        if not predictions:
            raise HTTPException(status_code=500, detail="No predictions generated")
        
        # Calculate ensemble prediction
        available_models = list(predictions.keys())
        total_score = sum(pred['prediction'] for pred in predictions.values())
        ensemble_score = total_score / len(available_models)
        ensemble_is_spam = ensemble_score > 0.5
        
        ensemble_prediction = {
            "score": ensemble_score,
            "is_spam": ensemble_is_spam,
            "confidence": max(ensemble_score, 1 - ensemble_score),
            "models_used": available_models,
            "agreement": sum(1 for pred in predictions.values() if pred['is_spam'])
        }
        
        response = PredictionResponse(
            timestamp=datetime.now().isoformat(),
            ensemble_prediction=ensemble_prediction,
            individual_predictions=predictions
        )
        
        # If email format requested, add headers and return modified email
        if request.return_format.lower() == "email":
            try:
                modified_email = email_processor.add_spam_headers(
                    parsed_email['original_message'], 
                    predictions
                )
                response.modified_email = modified_email
                response.email_headers_added = True
            except Exception as e:
                logger.warning(f"Could not add email headers: {e}")
                response.email_headers_added = False
        
        return response
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/predict/file")
async def predict_spam_file(file: UploadFile = File(...), return_format: str = Form(default="json")):
    """
    Upload and analyze an email file for spam detection
    """
    global model_manager, email_processor
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    if not model_manager.models:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    # Validate file type
    if not file.filename.endswith(('.eml', '.txt', '.msg')):
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .eml, .txt, or .msg files")
    
    try:
        # Read file content
        content = await file.read()
        email_content = content.decode('utf-8', errors='ignore')
        
        # Create request object
        request = EmailRequest(email_content=email_content, return_format=return_format)
        
        # Use the existing predict function
        result = await predict_spam(request)
        
        # If email format requested, return as plain text
        if return_format.lower() == "email" and result.modified_email:
            return PlainTextResponse(result.modified_email, media_type="text/plain")
        
        return result
        
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="Could not decode file content as UTF-8")
    except Exception as e:
        logger.error(f"File processing error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/predict/text")
async def predict_text_only(text: str = Form(...)):
    """
    Analyze plain text for spam detection (simplified endpoint)
    """
    global model_manager
    
    if not model_manager:
        raise HTTPException(status_code=503, detail="Model manager not initialized")
    
    if not model_manager.models:
        raise HTTPException(status_code=503, detail="No models loaded")
    
    try:
        # Create a simple email structure
        email_content = f"Subject: Text Analysis\nFrom: user@example.com\nTo: analysis@localhost\n\n{text}"
        
        # Create request object
        request = EmailRequest(email_content=email_content, return_format="json")
        
        # Use the existing predict function
        result = await predict_spam(request)
        
        # Return simplified response for text-only analysis
        return {
            "text": text,
            "ensemble_prediction": result.ensemble_prediction,
            "individual_predictions": result.individual_predictions,
            "timestamp": result.timestamp
        }
        
    except Exception as e:
        logger.error(f"Text analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.get("/explain/{model_name}")
async def get_model_explanation_info(model_name: str):
    """
    Get information about explanation methods for a specific model
    """
    explanation_info = {
        "bert": {
            "method": "Integrated Gradients",
            "description": "Computes attribution scores for input tokens using integrated gradients",
            "output": "Attribution scores for each token showing contribution to prediction"
        },
        "bilstm": {
            "method": "Attention Weights",
            "description": "Uses attention mechanism to show which tokens the model focuses on",
            "output": "Attention weights for each token indicating importance"
        },
        "cnn": {
            "method": "Grad-CAM",
            "description": "Gradient-weighted Class Activation Mapping for convolutional layers",
            "output": "Activation maps showing important regions in the input"
        }
    }
    
    if model_name.lower() not in explanation_info:
        raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
    
    return explanation_info[model_name.lower()]


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
