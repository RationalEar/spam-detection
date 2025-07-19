import os
import pickle
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import BertTokenizer
import logging

from models.bert import SpamBERT, tokenize_texts
from models.bilstm import BiLSTMSpam
from models.cnn import SpamCNN
from preprocess.preprocessor import preprocess_text
from utils.constants import DATA_PATH

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages loading and inference for all three spam detection models"""

    def __init__(self, model_paths: Dict[str, str], vocab_path: str = None, device: str = "auto"):
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        logger.info(f"Using device: {self.device}")

        self.models = {}
        self.model_paths = model_paths
        self.vocab_path = vocab_path
        self.failed_models = []

        # Initialize tokenizers and vocabulary
        self.bert_tokenizer = None
        self.vocab_to_idx = None
        self.idx_to_vocab = None
        self.embeddings = None

        self._load_models()


    def _load_vocabulary(self):
        """Load vocabulary for BiLSTM and CNN models"""
        try:
            if self.vocab_path and os.path.exists(self.vocab_path):
                with open(self.vocab_path, 'rb') as f:
                    vocab_data = pickle.load(f)
                    self.vocab_to_idx = vocab_data.get('vocab_to_idx', {})
                    self.idx_to_vocab = vocab_data.get('idx_to_vocab', {})
                    self.embeddings = vocab_data.get('embeddings')
                logger.info(f"Loaded vocabulary with {len(self.vocab_to_idx)} tokens")
            else:
                logger.warning(f"Vocabulary file not found at {self.vocab_path}. Creating minimal vocabulary.")
                # Create minimal vocabulary if not available
                self.vocab_to_idx = {'<PAD>': 0, '<UNK>': 1}
                self.idx_to_vocab = {0: '<PAD>', 1: '<UNK>'}
                # Create random embeddings as fallback
                vocab_size = 1000  # Default vocab size
                embedding_dim = 100
                self.embeddings = torch.randn(vocab_size, embedding_dim)

                # Expand vocab_to_idx to match embedding size
                for i in range(2, vocab_size):
                    self.vocab_to_idx[f'token_{i}'] = i
                    self.idx_to_vocab[i] = f'token_{i}'
        except Exception as e:
            logger.error(f"Error loading vocabulary: {e}")
            self._create_fallback_vocabulary()


    def _create_fallback_vocabulary(self):
        """Create fallback vocabulary when loading fails"""
        logger.info("Creating fallback vocabulary")
        vocab_size = 1000
        embedding_dim = 100

        self.vocab_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx_to_vocab = {0: '<PAD>', 1: '<UNK>'}

        for i in range(2, vocab_size):
            self.vocab_to_idx[f'token_{i}'] = i
            self.idx_to_vocab[i] = f'token_{i}'

        self.embeddings = torch.randn(vocab_size, embedding_dim)


    def _load_models(self):
        """Load all three models with improved error handling"""
        logger.info("Loading models...")

        # Load vocabulary first for BiLSTM and CNN
        self._load_vocabulary()

        # Load BERT model and tokenizer
        if 'bert' in self.model_paths:
            try:
                logger.info("Loading BERT model...")
                self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
                self.models['bert'] = SpamBERT()

                if os.path.exists(self.model_paths['bert']):
                    self.models['bert'].load(self.model_paths['bert'])
                    logger.info("BERT model loaded successfully")
                else:
                    logger.warning(f"BERT model file not found at {self.model_paths['bert']}. Model initialized with random weights.")

                self.models['bert'].to(self.device)
                self.models['bert'].eval()

            except Exception as e:
                logger.error(f"Failed to load BERT model: {e}")
                self.failed_models.append(('bert', str(e)))

        # Load BiLSTM model
        if 'bilstm' in self.model_paths:
            try:
                logger.info("Loading BiLSTM model...")
                vocab_size = len(self.vocab_to_idx)
                embedding_dim = self.embeddings.shape[1] if self.embeddings is not None else 100
                embeddings = self.embeddings if self.embeddings is not None else torch.randn(vocab_size, embedding_dim)

                self.models['bilstm'] = BiLSTMSpam(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    pretrained_embeddings=embeddings
                )

                if os.path.exists(self.model_paths['bilstm']):
                    self.models['bilstm'].load(self.model_paths['bilstm'], map_location=self.device)
                    logger.info("BiLSTM model loaded successfully")
                else:
                    logger.warning(f"BiLSTM model file not found at {self.model_paths['bilstm']}. Model initialized with random weights.")

                self.models['bilstm'].to(self.device)
                self.models['bilstm'].eval()

            except Exception as e:
                logger.error(f"Failed to load BiLSTM model: {e}")
                self.failed_models.append(('bilstm', str(e)))

        # Load CNN model
        if 'cnn' in self.model_paths:
            try:
                logger.info("Loading CNN model...")
                vocab_size = len(self.vocab_to_idx)
                embedding_dim = self.embeddings.shape[1] if self.embeddings is not None else 100
                embeddings = self.embeddings if self.embeddings is not None else torch.randn(vocab_size, embedding_dim)

                self.models['cnn'] = SpamCNN(
                    vocab_size=vocab_size,
                    embedding_dim=embedding_dim,
                    pretrained_embeddings=embeddings
                )

                if os.path.exists(self.model_paths['cnn']):
                    self.models['cnn'].load(self.model_paths['cnn'], map_location=self.device)
                    logger.info("CNN model loaded successfully")
                else:
                    logger.warning(f"CNN model file not found at {self.model_paths['cnn']}. Model initialized with random weights.")

                self.models['cnn'].to(self.device)
                self.models['cnn'].eval()

            except Exception as e:
                logger.error(f"Failed to load CNN model: {e}")
                self.failed_models.append(('cnn', str(e)))

        logger.info(f"Successfully loaded {len(self.models)} models")
        if self.failed_models:
            logger.warning(f"Failed to load {len(self.failed_models)} models: {self.failed_models}")


    def _text_to_indices(self, text: str, max_length: int = 512) -> torch.Tensor:
        """Convert text to token indices for BiLSTM and CNN models"""
        tokens = text.split()
        indices = []

        for token in tokens[:max_length]:
            idx = self.vocab_to_idx.get(token, self.vocab_to_idx.get('<UNK>', 1))
            indices.append(idx)

        # Pad or truncate to max_length
        if len(indices) < max_length:
            indices.extend([0] * (max_length - len(indices)))
        else:
            indices = indices[:max_length]

        return torch.tensor(indices, dtype=torch.long).unsqueeze(0)


    def predict_bert(self, text: str) -> Tuple[float, Dict]:
        """Make prediction with BERT model and get explanations"""
        if 'bert' not in self.models:
            return 0.0, {}

        # Tokenize text
        input_ids, attention_mask = tokenize_texts([text], self.bert_tokenizer, max_length=512)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Get prediction with attention weights
        with torch.no_grad():
            probs, attention_data = self.models['bert'](
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_attentions=True
            )
            prediction = float(probs[0].cpu().item())

        # Get integrated gradients explanation
        try:
            attributions, delta = self.models['bert'].compute_integrated_gradients(
                input_ids=input_ids,
                attention_mask=attention_mask,
                n_steps=20,  # Reduced for faster inference
                chunk_size=64
            )

            # Convert to meaningful explanation
            tokens = self.bert_tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
            attribution_scores = attributions[0].cpu().numpy()

            # Create token-attribution pairs
            explanations = []
            for i, (token, score) in enumerate(zip(tokens, attribution_scores)):
                if token not in ['[PAD]', '[CLS]', '[SEP]'] and attention_mask[0][i].item() == 1:
                    explanations.append({
                        'token': token,
                        'attribution': float(score),
                        'position': i
                    })

            explanation_data = {
                'method': 'integrated_gradients',
                'explanations': explanations,
                'convergence_delta': float(delta.cpu().item()),
                'attention_weights': attention_data if attention_data else {}
            }

        except Exception as e:
            logger.error(f"Error computing BERT explanations: {e}")
            explanation_data = {'method': 'integrated_gradients', 'error': str(e)}

        return prediction, explanation_data


    def predict_bilstm(self, text: str) -> Tuple[float, Dict]:
        """Make prediction with BiLSTM model and get attention explanations"""
        if 'bilstm' not in self.models:
            return 0.0, {}

        # Preprocess and convert to indices
        processed_text = preprocess_text(text)
        input_tensor = self._text_to_indices(processed_text, max_length=512)
        input_tensor = input_tensor.to(self.device)

        # Create attention mask
        attention_mask = (input_tensor != 0).float()

        # Get prediction and attention weights
        with torch.no_grad():
            prediction, attention_weights = self.models['bilstm'](input_tensor, attention_mask)
            prediction = float(prediction[0].cpu().item())

        # Create explanation from attention weights
        tokens = processed_text.split()[:512]  # Match max_length
        attention_scores = attention_weights[0].cpu().numpy()

        explanations = []
        for i, (token, score) in enumerate(zip(tokens, attention_scores)):
            if i < len(attention_scores):
                explanations.append({
                    'token': token,
                    'attention_weight': float(score),
                    'position': i
                })

        explanation_data = {
            'method': 'attention_weights',
            'explanations': explanations
        }

        return prediction, explanation_data


    def predict_cnn(self, text: str) -> Tuple[float, Dict]:
        """Make prediction with CNN model and get Grad-CAM explanations"""
        if 'cnn' not in self.models:
            return 0.0, {}

        # Preprocess and convert to indices
        processed_text = preprocess_text(text)
        input_tensor = self._text_to_indices(processed_text, max_length=512)
        input_tensor = input_tensor.to(self.device)

        # Get prediction
        with torch.no_grad():
            prediction = self.models['cnn'](input_tensor)
            prediction = float(prediction[0].cpu().item())

        # Get Grad-CAM explanation
        try:
            cam_maps = self.models['cnn'].grad_cam_auto(input_tensor)

            # Convert to meaningful explanation
            tokens = processed_text.split()[:512]  # Match max_length

            explanations = []
            if cam_maps is not None and len(cam_maps) > 0:
                # Use the first CAM map (usually the most relevant)
                cam_scores = cam_maps[0] if isinstance(cam_maps, list) else cam_maps
                if isinstance(cam_scores, torch.Tensor):
                    cam_scores = cam_scores.cpu().numpy()

                for i, (token, score) in enumerate(zip(tokens, cam_scores)):
                    if i < len(cam_scores):
                        explanations.append({
                            'token': token,
                            'grad_cam_score': float(score),
                            'position': i
                        })

            explanation_data = {
                'method': 'grad_cam',
                'explanations': explanations
            }

        except Exception as e:
            logger.error(f"Error computing CNN explanations: {e}")
            explanation_data = {'method': 'grad_cam', 'error': str(e)}

        return prediction, explanation_data


    def predict_all(self, text: str) -> Dict[str, Dict]:
        """Make predictions with all available models"""
        results = {}

        if 'bert' in self.models:
            pred, exp = self.predict_bert(text)
            results['bert'] = {
                'prediction': pred,
                'confidence': pred,
                'is_spam': pred > 0.5,
                'explanation': exp
            }

        if 'bilstm' in self.models:
            pred, exp = self.predict_bilstm(text)
            results['bilstm'] = {
                'prediction': pred,
                'confidence': pred,
                'is_spam': pred > 0.5,
                'explanation': exp
            }

        if 'cnn' in self.models:
            pred, exp = self.predict_cnn(text)
            results['cnn'] = {
                'prediction': pred,
                'confidence': pred,
                'is_spam': pred > 0.5,
                'explanation': exp
            }

        return results
