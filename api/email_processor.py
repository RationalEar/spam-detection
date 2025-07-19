import email
from email import policy
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage
import json
import re
from typing import Dict, Any, Optional
from datetime import datetime

from preprocess.email_parser import parse_email, extract_email_address, remove_server_signatures


class EmailProcessor:
    """Handles parsing and processing of email messages for spam detection"""
    
    def __init__(self):
        pass
    
    
    def parse_raw_email(self, raw_email: str) -> Dict[str, Any]:
        """
        Parse raw email content into structured data
        Args:
            raw_email: Raw email content as string
        Returns:
            Dictionary containing parsed email components
        """
        try:
            # Parse the email message
            msg = email.message_from_string(raw_email, policy=policy.default)
            
            # Extract headers
            subject = msg.get("Subject", "")
            sender = msg.get("From", "")
            sender_email = extract_email_address(sender)
            sender_domain = sender_email.split('@')[-1] if sender_email else ""
            reply_to = msg.get("Reply-To", "")
            reply_to_email = extract_email_address(reply_to)
            date_header = msg.get("Date", "")
            to_header = msg.get("To", "")
            cc_header = msg.get("Cc", "")
            
            # Extract body content
            body = ""
            html_body = ""
            
            if msg.is_multipart():
                for part in msg.walk():
                    content_type = part.get_content_type()
                    if content_type == "text/plain":
                        try:
                            content = part.get_payload(decode=True)
                            if content:
                                body = content.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
                    elif content_type == "text/html":
                        try:
                            content = part.get_payload(decode=True)
                            if content:
                                html_body = content.decode("utf-8", errors="ignore")
                        except Exception:
                            continue
            else:
                try:
                    content = msg.get_payload(decode=True)
                    if content:
                        body = content.decode("utf-8", errors="ignore")
                except Exception:
                    body = str(msg.get_payload())
            
            # Clean the body
            body = remove_server_signatures(body) if body else ""
            
            # Create combined text for analysis
            combined_text = f"<SUBJECT>{subject}</SUBJECT> <BODY>{body}</BODY>"
            
            return {
                "subject": subject,
                "sender": sender,
                "sender_email": sender_email,
                "sender_domain": sender_domain,
                "reply_to": reply_to_email,
                "date": date_header,
                "to": to_header,
                "cc": cc_header,
                "body": body,
                "html_body": html_body,
                "combined_text": combined_text,
                "original_message": msg
            }
            
        except Exception as e:
            raise ValueError(f"Error parsing email: {str(e)}")
    
    
    def add_spam_headers(self, original_msg: EmailMessage, predictions: Dict[str, Dict]) -> str:
        """
        Add spam detection results as headers to the email message
        Args:
            original_msg: Original email message object
            predictions: Dictionary containing predictions from all models
        Returns:
            Modified email as string with new headers
        """
        try:
            # Create a copy of the message to avoid modifying the original
            if hasattr(original_msg, 'as_string'):
                msg_str = original_msg.as_string()
            else:
                msg_str = str(original_msg)
            
            # Parse it again to get a fresh message object
            new_msg = email.message_from_string(msg_str, policy=policy.default)
            
            # Add timestamp header
            timestamp = datetime.now().isoformat()
            new_msg['X-Spam-Detection-Timestamp'] = timestamp
            
            # Calculate ensemble prediction
            available_models = list(predictions.keys())
            total_score = sum(pred['prediction'] for pred in predictions.values())
            ensemble_score = total_score / len(available_models) if available_models else 0.0
            ensemble_is_spam = ensemble_score > 0.5
            
            # Add ensemble headers
            new_msg['X-Spam-Score'] = f"{ensemble_score:.4f}"
            new_msg['X-Spam-Status'] = "SPAM" if ensemble_is_spam else "HAM"
            new_msg['X-Spam-Models-Used'] = ", ".join(available_models)
            
            # Add individual model predictions
            for model_name, result in predictions.items():
                model_prefix = f"X-Spam-{model_name.upper()}"
                new_msg[f'{model_prefix}-Score'] = f"{result['prediction']:.4f}"
                new_msg[f'{model_prefix}-Status'] = "SPAM" if result['is_spam'] else "HAM"
                
                # Add explanation summary
                explanation = result.get('explanation', {})
                if explanation and 'explanations' in explanation:
                    # Get top contributing tokens
                    explanations_list = explanation['explanations']
                    if explanations_list:
                        # Sort by importance (different fields for different models)
                        if model_name == 'bert':
                            sorted_exp = sorted(explanations_list, 
                                              key=lambda x: abs(x.get('attribution', 0)), 
                                              reverse=True)
                            score_field = 'attribution'
                        elif model_name == 'bilstm':
                            sorted_exp = sorted(explanations_list, 
                                              key=lambda x: x.get('attention_weight', 0), 
                                              reverse=True)
                            score_field = 'attention_weight'
                        elif model_name == 'cnn':
                            sorted_exp = sorted(explanations_list, 
                                              key=lambda x: abs(x.get('grad_cam_score', 0)), 
                                              reverse=True)
                            score_field = 'grad_cam_score'
                        else:
                            sorted_exp = explanations_list
                            score_field = 'score'
                        
                        # Get top 5 tokens
                        top_tokens = []
                        for exp in sorted_exp[:5]:
                            token = exp.get('token', '')
                            score = exp.get(score_field, 0)
                            if token and token not in ['[PAD]', '[CLS]', '[SEP]', '<PAD>', '<UNK>']:
                                top_tokens.append(f"{token}({score:.3f})")
                        
                        if top_tokens:
                            new_msg[f'{model_prefix}-Top-Features'] = ", ".join(top_tokens)
                
                # Add method information
                method = explanation.get('method', 'unknown')
                new_msg[f'{model_prefix}-Explanation-Method'] = method
            
            # Add detailed explanation as a JSON header (for programmatic access)
            explanation_json = json.dumps(predictions, indent=None, separators=(',', ':'))
            # Split long JSON into multiple headers if needed
            max_header_length = 998  # RFC 5322 recommends max 998 characters per line
            if len(explanation_json) > max_header_length:
                chunks = [explanation_json[i:i+max_header_length] 
                         for i in range(0, len(explanation_json), max_header_length)]
                for i, chunk in enumerate(chunks):
                    new_msg[f'X-Spam-Details-{i+1:02d}'] = chunk
            else:
                new_msg['X-Spam-Details'] = explanation_json
            
            # Add summary information
            spam_count = sum(1 for pred in predictions.values() if pred['is_spam'])
            total_count = len(predictions)
            new_msg['X-Spam-Agreement'] = f"{spam_count}/{total_count} models classify as spam"
            
            return new_msg.as_string()
            
        except Exception as e:
            raise ValueError(f"Error adding spam headers: {str(e)}")
    
    
    def create_response_email(self, predictions: Dict[str, Dict], 
                            original_subject: str = "") -> str:
        """
        Create a summary email with spam detection results
        Args:
            predictions: Dictionary containing predictions from all models
            original_subject: Subject of the original email
        Returns:
            Email message string with detection results
        """
        try:
            # Create new email message
            msg = EmailMessage()
            msg['Subject'] = f"Spam Detection Results: {original_subject}"
            msg['From'] = "spam-detection-api@localhost"
            msg['Date'] = datetime.now().strftime("%a, %d %b %Y %H:%M:%S %z")
            
            # Calculate ensemble results
            available_models = list(predictions.keys())
            total_score = sum(pred['prediction'] for pred in predictions.values())
            ensemble_score = total_score / len(available_models) if available_models else 0.0
            ensemble_is_spam = ensemble_score > 0.5
            
            # Create detailed report
            content = f"""Spam Detection Analysis Report
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

ENSEMBLE RESULTS:
- Overall Score: {ensemble_score:.4f}
- Classification: {"SPAM" if ensemble_is_spam else "HAM"}
- Models Used: {", ".join(available_models)}

INDIVIDUAL MODEL RESULTS:
"""
            
            for model_name, result in predictions.items():
                content += f"\n{model_name.upper()} Model:\n"
                content += f"  - Score: {result['prediction']:.4f}\n"
                content += f"  - Classification: {'SPAM' if result['is_spam'] else 'HAM'}\n"
                
                explanation = result.get('explanation', {})
                method = explanation.get('method', 'unknown')
                content += f"  - Explanation Method: {method}\n"
                
                if 'explanations' in explanation and explanation['explanations']:
                    content += "  - Top Contributing Features:\n"
                    explanations_list = explanation['explanations']
                    
                    # Sort by importance
                    if model_name == 'bert':
                        sorted_exp = sorted(explanations_list, 
                                          key=lambda x: abs(x.get('attribution', 0)), 
                                          reverse=True)
                        score_field = 'attribution'
                    elif model_name == 'bilstm':
                        sorted_exp = sorted(explanations_list, 
                                          key=lambda x: x.get('attention_weight', 0), 
                                          reverse=True)
                        score_field = 'attention_weight'
                    elif model_name == 'cnn':
                        sorted_exp = sorted(explanations_list, 
                                          key=lambda x: abs(x.get('grad_cam_score', 0)), 
                                          reverse=True)
                        score_field = 'grad_cam_score'
                    else:
                        sorted_exp = explanations_list
                        score_field = 'score'
                    
                    # Show top 10 features
                    for i, exp in enumerate(sorted_exp[:10]):
                        token = exp.get('token', '')
                        score = exp.get(score_field, 0)
                        if token and token not in ['[PAD]', '[CLS]', '[SEP]', '<PAD>', '<UNK>']:
                            content += f"    {i+1}. {token}: {score:.4f}\n"
                
                if 'error' in explanation:
                    content += f"  - Error: {explanation['error']}\n"
            
            content += f"\n\nNote: This analysis was performed automatically. "
            content += f"Scores range from 0.0 (definitely ham) to 1.0 (definitely spam)."
            
            msg.set_content(content)
            
            return msg.as_string()
            
        except Exception as e:
            raise ValueError(f"Error creating response email: {str(e)}")
