#!/usr/bin/env python3
"""
Comprehensive test script for the Spam Detection API
"""

import os
import sys
import requests
import json
import time
import tempfile
from typing import Dict, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.client_example import SpamDetectionClient, create_sample_emails


class APITester:
    """Comprehensive API testing class"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.client = SpamDetectionClient(base_url)
        self.base_url = base_url
        self.test_results = {}
        self.failed_tests = []
    
    
    def wait_for_api(self, max_attempts: int = 30, delay: int = 2) -> bool:
        """Wait for API to become available"""
        print(f"Waiting for API at {self.base_url}...")
        
        for attempt in range(max_attempts):
            try:
                response = requests.get(f"{self.base_url}/health", timeout=5)
                if response.status_code == 200:
                    print(f"API is ready after {attempt + 1} attempts")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            if attempt < max_attempts - 1:
                print(f"Attempt {attempt + 1}/{max_attempts} failed, waiting {delay}s...")
                time.sleep(delay)
        
        print(f"API not available after {max_attempts} attempts")
        return False
    
    
    def run_test(self, test_name: str, test_func) -> bool:
        """Run a single test and record results"""
        print(f"\n{'='*50}")
        print(f"Running test: {test_name}")
        print(f"{'='*50}")
        
        try:
            result = test_func()
            self.test_results[test_name] = {"status": "PASSED", "result": result}
            print(f"✅ {test_name} PASSED")
            return True
        except Exception as e:
            self.test_results[test_name] = {"status": "FAILED", "error": str(e)}
            self.failed_tests.append(test_name)
            print(f"❌ {test_name} FAILED: {e}")
            return False
    
    
    def test_health_check(self) -> Dict[str, Any]:
        """Test health check endpoint"""
        response = self.client.health_check()
        assert "status" in response
        assert response["status"] == "healthy"
        assert "models_loaded" in response
        print(f"Health check passed. Models loaded: {response['models_loaded']}")
        return response
    
    
    def test_models_status(self) -> Dict[str, Any]:
        """Test models status endpoint"""
        response = self.client.get_models_status()
        assert "total_models" in response
        assert "loaded_models" in response
        print(f"Models status: {response['total_models']} total, {response['loaded_models']} loaded")
        return response
    
    
    def test_root_endpoint(self) -> Dict[str, Any]:
        """Test root endpoint"""
        response = requests.get(f"{self.base_url}/")
        response.raise_for_status()
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        print(f"Root endpoint returned version {data['version']}")
        return data
    
    
    def test_spam_email_prediction(self) -> Dict[str, Any]:
        """Test spam email prediction"""
        spam_email, _ = create_sample_emails()
        result = self.client.predict_email(spam_email)
        
        assert "ensemble_prediction" in result
        assert "individual_predictions" in result
        assert "timestamp" in result
        
        ensemble = result["ensemble_prediction"]
        assert "score" in ensemble
        assert "is_spam" in ensemble
        assert isinstance(ensemble["score"], float)
        
        print(f"Spam email prediction: {ensemble['score']:.3f} ({'SPAM' if ensemble['is_spam'] else 'HAM'})")
        
        # Check individual model predictions
        for model, pred in result["individual_predictions"].items():
            print(f"  {model.upper()}: {pred['prediction']:.3f} ({'SPAM' if pred['is_spam'] else 'HAM'})")
            assert "explanation" in pred
        
        return result
    
    
    def test_ham_email_prediction(self) -> Dict[str, Any]:
        """Test ham email prediction"""
        _, ham_email = create_sample_emails()
        result = self.client.predict_email(ham_email)
        
        assert "ensemble_prediction" in result
        assert "individual_predictions" in result
        
        ensemble = result["ensemble_prediction"]
        print(f"Ham email prediction: {ensemble['score']:.3f} ({'SPAM' if ensemble['is_spam'] else 'HAM'})")
        
        return result
    
    
    def test_email_format_output(self) -> Dict[str, Any]:
        """Test email format output with headers"""
        spam_email, _ = create_sample_emails()
        result = self.client.predict_email(spam_email, return_format="email")
        
        if "modified_email" in result:
            modified_email = result["modified_email"]
            assert "X-Spam-Score" in modified_email
            assert "X-Spam-Status" in modified_email
            print("Email headers added successfully")
            
            # Count spam headers
            spam_headers = [line for line in modified_email.split('\n') if line.startswith('X-Spam')]
            print(f"Added {len(spam_headers)} spam detection headers")
        else:
            print("Warning: Email format output not available")
        
        return result
    
    
    def test_text_prediction(self) -> Dict[str, Any]:
        """Test plain text prediction"""
        test_text = "Get rich quick! Click here now for amazing offers! Free money!"
        result = self.client.predict_text(test_text)
        
        assert "ensemble_prediction" in result
        assert "individual_predictions" in result
        assert "text" in result
        
        ensemble = result["ensemble_prediction"]
        print(f"Text prediction: {ensemble['score']:.3f} ({'SPAM' if ensemble['is_spam'] else 'HAM'})")
        
        return result
    
    
    def test_file_upload(self) -> Dict[str, Any]:
        """Test file upload functionality"""
        spam_email, _ = create_sample_emails()
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.eml', delete=False) as f:
            f.write(spam_email)
            temp_file = f.name
        
        try:
            result = self.client.predict_file(temp_file)
            assert "ensemble_prediction" in result
            assert "individual_predictions" in result
            
            ensemble = result["ensemble_prediction"]
            print(f"File upload prediction: {ensemble['score']:.3f} ({'SPAM' if ensemble['is_spam'] else 'HAM'})")
            
            return result
        finally:
            os.unlink(temp_file)
    
    
    def test_explanation_endpoints(self) -> Dict[str, Any]:
        """Test explanation information endpoints"""
        results = {}
        
        for model in ['bert', 'bilstm', 'cnn']:
            try:
                info = self.client.get_explanation_info(model)
                assert "method" in info
                assert "description" in info
                print(f"{model.upper()}: {info['method']}")
                results[model] = info
            except requests.exceptions.RequestException as e:
                print(f"Warning: Could not get explanation info for {model}: {e}")
        
        return results
    
    
    def test_error_handling(self) -> Dict[str, Any]:
        """Test error handling for invalid requests"""
        results = {}
        
        # Test invalid email content
        try:
            self.client.predict_email("")
            results["empty_email"] = "FAILED - Should have returned error"
        except requests.exceptions.RequestException:
            results["empty_email"] = "PASSED - Correctly handled empty email"
            print("✓ Empty email correctly rejected")
        
        # Test invalid model explanation
        try:
            response = requests.get(f"{self.base_url}/explain/invalid_model")
            if response.status_code == 404:
                results["invalid_model"] = "PASSED - Correctly returned 404"
                print("✓ Invalid model correctly rejected")
            else:
                results["invalid_model"] = f"FAILED - Returned {response.status_code}"
        except requests.exceptions.RequestException as e:
            results["invalid_model"] = f"FAILED - Exception: {e}"
        
        return results
    
    
    def test_performance(self) -> Dict[str, Any]:
        """Test API performance with multiple requests"""
        _, ham_email = create_sample_emails()
        
        num_requests = 5
        times = []
        
        print(f"Testing performance with {num_requests} requests...")
        
        for i in range(num_requests):
            start_time = time.time()
            try:
                result = self.client.predict_email(ham_email)
                end_time = time.time()
                request_time = end_time - start_time
                times.append(request_time)
                print(f"Request {i+1}: {request_time:.2f}s")
            except Exception as e:
                print(f"Request {i+1} failed: {e}")
        
        if times:
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
            performance_results = {
                "average_time": avg_time,
                "min_time": min_time,
                "max_time": max_time,
                "successful_requests": len(times),
                "total_requests": num_requests
            }
            
            print(f"Performance summary:")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Min: {min_time:.2f}s")
            print(f"  Max: {max_time:.2f}s")
            print(f"  Success rate: {len(times)}/{num_requests}")
            
            return performance_results
        else:
            raise Exception("All performance test requests failed")
    
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all tests and return summary"""
        print("🚀 Starting comprehensive API tests...")
        
        # Wait for API to be ready
        if not self.wait_for_api():
            print("❌ API not available. Make sure it's running with: ./start_api.sh")
            return {"error": "API not available"}
        
        # Define tests to run
        tests = [
            ("Health Check", self.test_health_check),
            ("Models Status", self.test_models_status),
            ("Root Endpoint", self.test_root_endpoint),
            ("Spam Email Prediction", self.test_spam_email_prediction),
            ("Ham Email Prediction", self.test_ham_email_prediction),
            ("Email Format Output", self.test_email_format_output),
            ("Text Prediction", self.test_text_prediction),
            ("File Upload", self.test_file_upload),
            ("Explanation Endpoints", self.test_explanation_endpoints),
            ("Error Handling", self.test_error_handling),
            ("Performance Test", self.test_performance)
        ]
        
        # Run all tests
        passed = 0
        total = len(tests)
        
        for test_name, test_func in tests:
            if self.run_test(test_name, test_func):
                passed += 1
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"TEST SUMMARY")
        print(f"{'='*60}")
        print(f"Total tests: {total}")
        print(f"Passed: {passed}")
        print(f"Failed: {total - passed}")
        print(f"Success rate: {passed/total*100:.1f}%")
        
        if self.failed_tests:
            print(f"\nFailed tests:")
            for test in self.failed_tests:
                print(f"  ❌ {test}")
        
        return {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": passed/total*100,
            "failed_tests": self.failed_tests,
            "detailed_results": self.test_results
        }


def main():
    """Main test function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Spam Detection API")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--output", help="Save detailed results to JSON file")
    parser.add_argument("--test", help="Run specific test only")
    
    args = parser.parse_args()
    
    tester = APITester(args.url)
    
    if args.test:
        # Run specific test
        test_methods = {
            "health": tester.test_health_check,
            "models": tester.test_models_status,
            "spam": tester.test_spam_email_prediction,
            "ham": tester.test_ham_email_prediction,
            "text": tester.test_text_prediction,
            "file": tester.test_file_upload,
            "explain": tester.test_explanation_endpoints,
            "error": tester.test_error_handling,
            "performance": tester.test_performance
        }
        
        if args.test in test_methods:
            tester.run_test(args.test.title(), test_methods[args.test])
        else:
            print(f"Unknown test: {args.test}")
            print(f"Available tests: {list(test_methods.keys())}")
            return 1
    else:
        # Run all tests
        results = tester.run_all_tests()
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nDetailed results saved to {args.output}")
    
    return 0 if not tester.failed_tests else 1


if __name__ == "__main__":
    sys.exit(main())
