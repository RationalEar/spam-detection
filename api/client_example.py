import requests
import json
import os
from typing import Dict, Any


class SpamDetectionClient:
    """Client for interacting with the Spam Detection API"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()


    def health_check(self) -> Dict[str, Any]:
        """Check API health status"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()


    def get_models_status(self) -> Dict[str, Any]:
        """Get status of loaded models"""
        response = self.session.get(f"{self.base_url}/models/status")
        response.raise_for_status()
        return response.json()


    def predict_email(self, email_content: str, return_format: str = "json") -> Dict[str, Any]:
        """
        Predict spam for email content
        Args:
            email_content: Raw email content as string
            return_format: 'json' or 'email'
        Returns:
            Prediction results
        """
        payload = {
            "email_content": email_content,
            "return_format": return_format
        }
        response = self.session.post(f"{self.base_url}/predict", json=payload)
        response.raise_for_status()

        if return_format == "email":
            return {"modified_email": response.text}
        return response.json()


    def predict_file(self, file_path: str, return_format: str = "json") -> Dict[str, Any]:
        """
        Upload and predict spam for email file
        Args:
            file_path: Path to email file
            return_format: 'json' or 'email'
        Returns:
            Prediction results
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        with open(file_path, 'rb') as f:
            files = {'file': (os.path.basename(file_path), f, 'text/plain')}
            data = {'return_format': return_format}
            response = self.session.post(f"{self.base_url}/predict/file", files=files, data=data)

        response.raise_for_status()

        if return_format == "email":
            return {"modified_email": response.text}
        return response.json()


    def predict_text(self, text: str) -> Dict[str, Any]:
        """
        Predict spam for plain text
        Args:
            text: Plain text to analyze
        Returns:
            Prediction results
        """
        data = {'text': text}
        response = self.session.post(f"{self.base_url}/predict/text", data=data)
        response.raise_for_status()
        return response.json()


    def get_explanation_info(self, model_name: str) -> Dict[str, Any]:
        """
        Get explanation method information for a model
        Args:
            model_name: Name of the model ('bert', 'bilstm', 'cnn')
        Returns:
            Explanation method information
        """
        response = self.session.get(f"{self.base_url}/explain/{model_name}")
        response.raise_for_status()
        return response.json()


def create_sample_emails():
    """Create sample email content for testing"""

    spam_email = """From: winner@lottery-international.com
To: you@example.com
Subject: CONGRATULATIONS!!! YOU'VE WON $1,000,000!!!
Date: Mon, 01 Jan 2024 12:00:00 +0000

CONGRATULATIONS!!!

You have been selected as the WINNER of our international lottery!
You've won ONE MILLION DOLLARS ($1,000,000)!!!

To claim your prize, simply reply with:
- Your full name
- Your bank account details
- Your social security number

This offer expires in 24 hours! Act now!

Click here to claim: http://suspicious-lottery.fake

Best regards,
International Lottery Commission
"""

    ham_email = """From: john.smith@company.com
To: team@company.com
Subject: Team Meeting Tomorrow at 2 PM
Date: Mon, 01 Jan 2024 10:30:00 +0000

Hi Team,

Just a reminder that we have our weekly team meeting tomorrow at 2 PM in the conference room.

Agenda:
1. Project updates
2. Q1 planning
3. Any other business

Please let me know if you can't attend.

Best regards,
John Smith
Project Manager
"""

    return spam_email, ham_email


def main():
    """Example usage of the Spam Detection API client"""

    client = SpamDetectionClient()

    print("=== Spam Detection API Client Example ===\n")

    # Check API health
    try:
        health = client.health_check()
        print("API Health:", json.dumps(health, indent=2))
    except requests.exceptions.RequestException as e:
        print(f"API not available: {e}")
        print("Make sure the API is running with: ./start_api.sh")
        return

    # Check models status
    try:
        models_status = client.get_models_status()
        print("\nModels Status:", json.dumps(models_status, indent=2))
    except requests.exceptions.RequestException as e:
        print(f"Error getting models status: {e}")
        return

    # Get sample emails
    spam_email, ham_email = create_sample_emails()

    # Test spam email
    print("\n=== Testing Spam Email ===")
    try:
        result = client.predict_email(spam_email)
        print("Ensemble Prediction:", result['ensemble_prediction'])
        print("Individual Predictions:")
        for model, pred in result['individual_predictions'].items():
            print(f"  {model.upper()}: {pred['prediction']:.4f} ({'SPAM' if pred['is_spam'] else 'HAM'})")

            # Show top explanations
            if 'explanation' in pred and 'explanations' in pred['explanation']:
                explanations = pred['explanation']['explanations']
                if explanations:
                    print(f"    Top features: {explanations[:3]}")
    except requests.exceptions.RequestException as e:
        print(f"Error predicting spam email: {e}")

    # Test ham email
    print("\n=== Testing Ham Email ===")
    try:
        result = client.predict_email(ham_email)
        print("Ensemble Prediction:", result['ensemble_prediction'])
        print("Individual Predictions:")
        for model, pred in result['individual_predictions'].items():
            print(f"  {model.upper()}: {pred['prediction']:.4f} ({'SPAM' if pred['is_spam'] else 'HAM'})")
    except requests.exceptions.RequestException as e:
        print(f"Error predicting ham email: {e}")

    # Test email format output
    print("\n=== Testing Email Format Output ===")
    try:
        result = client.predict_email(spam_email, return_format="email")
        if 'modified_email' in result:
            lines = result['modified_email'].split('\n')
            print("Modified email headers:")
            for line in lines[:20]:  # Show first 20 lines
                if line.startswith('X-Spam'):
                    print(f"  {line}")
    except requests.exceptions.RequestException as e:
        print(f"Error getting email format: {e}")

    # Test plain text prediction
    print("\n=== Testing Plain Text Prediction ===")
    try:
        test_text = "Get rich quick! Click here now for amazing offers! Free money!"
        result = client.predict_text(test_text)
        print(f"Text: {test_text}")
        print("Ensemble Prediction:", result['ensemble_prediction'])
    except requests.exceptions.RequestException as e:
        print(f"Error predicting text: {e}")

    # Get explanation information
    print("\n=== Model Explanation Methods ===")
    for model in ['bert', 'bilstm', 'cnn']:
        try:
            info = client.get_explanation_info(model)
            print(f"{model.upper()}: {info['method']} - {info['description']}")
        except requests.exceptions.RequestException as e:
            print(f"Error getting {model} explanation info: {e}")


if __name__ == "__main__":
    main()
