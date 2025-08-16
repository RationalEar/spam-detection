#!/usr/bin/env python3
"""
Script to analyze top 20 influential words from test dataset using BERT Layer Integrated Gradients
"""
import os
import sys
import torch
import pandas as pd
from transformers import BertTokenizer

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.bert import SpamBERT
from explainability.BertExplanationMetrics import analyze_test_dataset_influential_words
from utils.constants import DATA_PATH, MODEL_SAVE_PATH


def main():
    """
    Main function to analyze influential words in test dataset
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load test data
    print("Loading test dataset...")
    test_df = pd.read_pickle(os.path.join(DATA_PATH, 'data/processed/test.pkl'))
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()
    
    print(f"Loaded {len(test_texts)} test samples")
    print(f"Spam samples: {sum(test_labels)}")
    print(f"Ham samples: {len(test_labels) - sum(test_labels)}")
    
    # Initialize BERT tokenizer and model
    print("\nInitializing BERT model...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SpamBERT(dropout=0.2)
    
    # Load trained model weights
    model_path = os.path.join(MODEL_SAVE_PATH, 'spam_bert_final.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    print("Model loaded successfully!")
    
    # Analyze influential words using Layer Integrated Gradients
    print("\n" + "="*60)
    print("ANALYZING TOP 20 INFLUENTIAL WORDS USING LAYER INTEGRATED GRADIENTS")
    print("="*60)
    
    results = analyze_test_dataset_influential_words(
        model=model,
        tokenizer=tokenizer,
        test_texts=test_texts,
        test_labels=test_labels,
        device=device,
        top_k=20,
        method='integrated_gradients'
    )
    
    # Save results
    results_dir = os.path.join(DATA_PATH, 'results', 'influential_words')
    os.makedirs(results_dir, exist_ok=True)
    
    # Save detailed results to CSV
    results_file = os.path.join(results_dir, 'bert_top_influential_words.csv')
    
    # Create DataFrames for different categories
    overall_df = pd.DataFrame(results['top_overall_words'])
    spam_df = pd.DataFrame(results['top_spam_words'])
    ham_df = pd.DataFrame(results['top_ham_words'])
    discriminative_df = pd.DataFrame(results['top_discriminative_words'])
    
    # Save to separate sheets or files
    overall_df.to_csv(os.path.join(results_dir, 'top_overall_words.csv'), index=False)
    spam_df.to_csv(os.path.join(results_dir, 'top_spam_words.csv'), index=False)
    ham_df.to_csv(os.path.join(results_dir, 'top_ham_words.csv'), index=False)
    discriminative_df.to_csv(os.path.join(results_dir, 'top_discriminative_words.csv'), index=False)
    
    print(f"\nResults saved to: {results_dir}")
    print("\nFiles created:")
    print("- top_overall_words.csv: Most influential words across all samples")
    print("- top_spam_words.csv: Most influential words in spam emails")
    print("- top_ham_words.csv: Most influential words in ham emails")
    print("- top_discriminative_words.csv: Words that best distinguish spam from ham")
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print("SUMMARY STATISTICS")
    print(f"{'='*60}")
    print(f"Total texts processed: {results['total_texts_processed']}")
    print(f"Total unique influential words: {results['statistics']['total_unique_words']}")
    print(f"Unique words in spam: {results['statistics']['spam_unique_words']}")
    print(f"Unique words in ham: {results['statistics']['ham_unique_words']}")
    
    return results


def analyze_single_text_example():
    """
    Example function to analyze a single text for influential words
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = SpamBERT(dropout=0.2)
    model_path = os.path.join(MODEL_SAVE_PATH, 'spam_bert_final.pt')
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Import the BertExplanationMetrics class
    from explainability.BertExplanationMetrics import BertExplanationMetrics
    
    # Initialize analyzer
    analyzer = BertExplanationMetrics(model, tokenizer, device)
    
    # Example text
    example_text = "URGENT! You have won $1000000! Click here to claim your prize now! Limited time offer!"
    
    print("Analyzing single text example:")
    print(f"Text: {example_text}")
    print("\nTop 10 influential words:")
    
    # Get top influential words
    influential_words = analyzer.get_top_influential_words(example_text, top_k=10, method='integrated_gradients')
    
    print(f"{'Rank':<4} {'Word':<15} {'Token':<15} {'Score':<10}")
    print("-" * 50)
    for word_info in influential_words:
        print(f"{word_info['rank']:<4} {word_info['word']:<15} {word_info['token']:<15} {word_info['importance_score']:<10.4f}")


if __name__ == "__main__":
    # Run main analysis
    results = main()
    
    # Run single text example
    print("\n" + "="*60)
    print("SINGLE TEXT EXAMPLE")
    print("="*60)
    analyze_single_text_example()
