import os

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from models.bert import SpamBERT
from models.bilstm import BiLSTMSpam
from models.cnn import SpamCNN
from utils.functions import build_vocab, encode


def train_bilstm(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, max_norm=1.0,
                 adversarial_training=True, epsilon=0.1, model_save_path=''):
    """
    Training loop for BiLSTM model with gradient clipping and adversarial training
    Args:
        model: BiLSTM model instance
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        criterion: Loss function
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
        device: Device to train on
        max_norm: Maximum gradient norm for clipping
        adversarial_training: Whether to use adversarial training
        epsilon: Epsilon for adversarial example generation
        model_save_path: Directory to save model checkpoints
    """
    best_val_loss = float('inf')

    # Create save directory if it doesn't exist
    if model_save_path:
        os.makedirs(model_save_path, exist_ok=True)

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        correct_train = 0
        total_train = 0

        for batch in train_loader:
            inputs, labels = [b.to(device) for b in batch]
            batch_size = inputs.size(0)

            # Regular forward pass
            optimizer.zero_grad()
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)

            # Adversarial training
            if adversarial_training:
                # Generate adversarial examples
                with torch.set_grad_enabled(True):
                    adv_embeddings = model.generate_adversarial_example(inputs, labels, epsilon=epsilon)
                    # Forward pass with adversarial examples
                    adv_outputs, _ = model(adv_embeddings)
                    adv_loss = criterion(adv_outputs, labels)
                    # Combine losses
                    loss = 0.5 * (loss + adv_loss)

            loss.backward()
            # Clip gradients
            model.clip_gradients(max_norm)
            optimizer.step()

            total_train_loss += loss.item() * batch_size
            predicted = (outputs > 0.5).float()
            correct_train += (predicted == labels).sum().item()
            total_train += batch_size

        avg_train_loss = total_train_loss / total_train
        train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        total_val_loss = 0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = [b.to(device) for b in batch]
                batch_size = inputs.size(0)

                outputs, _ = model(inputs)
                loss = criterion(outputs, labels)

                total_val_loss += loss.item() * batch_size
                predicted = (outputs > 0.5).float()
                correct_val += (predicted == labels).sum().item()
                total_val += batch_size

        avg_val_loss = total_val_loss / total_val
        val_acc = correct_val / total_val

        # Update training history
        model.update_training_history(
            train_loss=avg_train_loss,
            val_loss=avg_val_loss,
            train_acc=train_acc,
            val_acc=val_acc
        )

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            if model_save_path:
                best_model_path = os.path.join(model_save_path, 'best_bilstm_model.pt')
                model.save(best_model_path)
                print(f"Saved best model to {best_model_path}")

    return model


def train_model(model_type, train_df, val_df=None, test_df=None, embedding_dim=300, pretrained_embeddings=None,
                model_save_path='', max_len=200, evaluate=False, early_stopping_patience=5):
    """
    Train a model with support for 80/10/10 train/val/test split
    
    Args:
        model_type: Type of model to train ('cnn', 'bilstm', or 'bert')
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data (if None, will use test_df for validation)
        test_df: DataFrame containing test data
        embedding_dim: Dimension of word embeddings
        pretrained_embeddings: Pre-trained embeddings tensor
        model_save_path: Directory to save model checkpoints
        max_len: Maximum sequence length
        evaluate: Whether to evaluate the model after training
        early_stopping_patience: Number of epochs with no improvement after which training will be stopped
    """
    # Check if we have all three splits
    has_three_way_split = val_df is not None and test_df is not None
    
    if not has_three_way_split:
        if test_df is None:
            raise ValueError("test_df must be provided")
        # Fall back to two-way split if validation set not provided
        print("Warning: No validation set provided. Using test set for validation.")
        val_df = test_df
    
    # Build vocabulary from training data only
    word2idx, idx2word = build_vocab(train_df['text'])

    # Encode all datasets
    X_train = torch.tensor([encode(t, word2idx, max_len) for t in train_df['text']])
    y_train = torch.tensor(train_df['label'].values, dtype=torch.float32)
    
    X_val = torch.tensor([encode(t, word2idx, max_len) for t in val_df['text']])
    y_val = torch.tensor(val_df['label'].values, dtype=torch.float32)
    
    if has_three_way_split:
        X_test = torch.tensor([encode(t, word2idx, max_len) for t in test_df['text']])
        y_test = torch.tensor(test_df['label'].values, dtype=torch.float32)
    else:
        X_test, y_test = X_val, y_val

    # Choose model: 'cnn', 'bilstm', or 'bert'
    if model_type == 'cnn':
        model = SpamCNN(vocab_size=len(word2idx), embedding_dim=embedding_dim,
                        pretrained_embeddings=pretrained_embeddings)
        train_inputs, train_labels = X_train, y_train
        val_inputs, val_labels = X_val, y_val
        test_inputs, test_labels = X_test, y_test
    elif model_type == 'bilstm':
        model = BiLSTMSpam(vocab_size=len(word2idx), embedding_dim=embedding_dim,
                           pretrained_embeddings=pretrained_embeddings)
        train_inputs, train_labels = X_train, y_train
        val_inputs, val_labels = X_val, y_val
        test_inputs, test_labels = X_test, y_test
    elif model_type == 'bert':
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # Tokenize with BERT tokenizer

        def bert_encode(texts, tokenizer, max_len=200):  # Tokenize and encode sequences
            return tokenizer(texts.tolist(), padding='max_length', truncation=True, max_length=max_len,
                             return_tensors='pt')

        train_encodings = bert_encode(train_df['text'], tokenizer, max_len)
        val_encodings = bert_encode(val_df['text'], tokenizer, max_len)
        test_encodings = bert_encode(test_df['text'], tokenizer, max_len)
        
        model = SpamBERT()
        train_inputs, train_labels = train_encodings, y_train
        val_inputs, val_labels = val_encodings, y_val
        test_inputs, test_labels = test_encodings, y_test
    else:
        raise ValueError('Invalid model_type')

    # Set model-specific training parameters
    if model_type == 'cnn':
        epochs = 50
        learning_rate = 1e-3
    elif model_type == 'bilstm':
        epochs = 40
        learning_rate = 8e-4
    elif model_type == 'bert':
        epochs = 10
        learning_rate = 2e-5
    else:
        raise ValueError('Invalid model_type')

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    batch_size = 32
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    if model_type == 'bilstm':
        # Create data loaders for BiLSTM
        train_dataset = TensorDataset(train_inputs, train_labels)
        val_dataset = TensorDataset(val_inputs, val_labels)
        test_dataset = TensorDataset(test_inputs, test_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Train BiLSTM with specialized training loop
        model = train_bilstm(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            num_epochs=epochs,
            device=device,
            max_norm=1.0,
            adversarial_training=True,
            epsilon=0.1,
            model_save_path=model_save_path
        )
    else:  # CNN or BERT
        # Create datasets
        if model_type == 'bert':
            train_dataset = TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
            val_dataset = TensorDataset(val_inputs['input_ids'], val_inputs['attention_mask'], val_labels)
            test_dataset = TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
        else:  # CNN
            train_dataset = TensorDataset(train_inputs, train_labels)
            val_dataset = TensorDataset(val_inputs, val_labels)
            test_dataset = TensorDataset(test_inputs, test_labels)

        # Create dataloaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)

        # Standard training loop for CNN and BERT with early stopping
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            train_steps = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                
                if model_type == 'bert':
                    input_ids, attention_mask, labels = [b.to(device) for b in batch]
                    outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                else:  # CNN
                    inputs, labels = [b.to(device) for b in batch]
                    outputs = model(inputs)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]

                # Check tensor dimensions before squeezing
                if outputs.dim() > 1 and outputs.shape[1] == 1:
                    outputs = outputs.squeeze(1)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                total_train_loss += loss.item()
                train_steps += 1
            
            avg_train_loss = total_train_loss / train_steps
            
            # Validation phase
            model.eval()
            total_val_loss = 0
            val_steps = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if model_type == 'bert':
                        input_ids, attention_mask, labels = [b.to(device) for b in batch]
                        outputs, _ = model(input_ids=input_ids, attention_mask=attention_mask)
                    else:  # CNN
                        inputs, labels = [b.to(device) for b in batch]
                        outputs = model(inputs)
                        if isinstance(outputs, tuple):
                            outputs = outputs[0]
                    
                    # Check tensor dimensions before squeezing
                    if outputs.dim() > 1 and outputs.shape[1] == 1:
                        outputs = outputs.squeeze(1)
                    
                    loss = criterion(outputs, labels)
                    total_val_loss += loss.item()
                    val_steps += 1
            
            avg_val_loss = total_val_loss / val_steps
            
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Early stopping check
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                
                # Save best model checkpoint
                if model_save_path:
                    os.makedirs(model_save_path, exist_ok=True)
                    best_model_path = os.path.join(model_save_path, f'best_{model_type}_model.pt')
                    torch.save(best_model_state, best_model_path)
                    print(f"Saved best model to {best_model_path}")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break
        
        # Load best model for final save and evaluation
        if best_model_state is not None:
            model.load_state_dict(best_model_state)

    # Save final model to specified path
    model_save_file = os.path.join(model_save_path, f'spam_{model_type}_final.pt')
    model.save(model_save_file)
    print(f"Final model saved to {model_save_file}")

    if evaluate:
        print("\nEvaluating on test set:")
        if model_type == 'bert':
            evaluate_model(model, model_type, test_loader=test_loader)
        else:
            evaluate_model(model, model_type, X_test=test_inputs, y_test=test_labels)

    return model


def evaluate_model(model, model_type, X_test=None, y_test=None, test_loader=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    if model_type == 'bert':
        y_pred, y_true = [], []
        for batch in test_loader:
            batch = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1]}
                outputs = model(**inputs)

            # Fix: Handle tuple output from SpamBERT model
            if isinstance(outputs, tuple):
                probs = outputs[0]  # First element contains the probabilities
            else:
                probs = outputs

            predictions = (probs > 0.5).long().cpu().numpy()
            y_pred.extend(predictions)
            y_true.extend(batch[2].cpu().numpy())

        print(classification_report(y_true, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

    elif model_type in ['cnn', 'bilstm']:
        X_test = X_test.to(device)
        with torch.no_grad():
            outputs = model(X_test)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            # Check tensor dimensions before squeezing
            if outputs.dim() > 1 and outputs.shape[1] == 1:
                outputs = outputs.squeeze(1)
            predictions = (outputs > 0.5).cpu().numpy()

        print(classification_report(y_test.numpy(), predictions))
        print("Confusion Matrix:\n", confusion_matrix(y_test.numpy(), predictions))
