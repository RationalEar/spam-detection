import os
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from models.bilstm import BiLSTMSpam
from utils.functions import build_vocab, encode


def train_bilstm(train_df, val_df, test_df, embedding_dim=300, pretrained_embeddings=None,
                 model_save_path='', max_len=200, max_norm=1.0, adversarial_training=True, epsilon=0.1):
    """
    Train BiLSTM model with gradient clipping and adversarial training

    Args:
        train_df: DataFrame containing training data
        val_df: DataFrame containing validation data
        test_df: DataFrame containing test data
        embedding_dim: Dimension of word embeddings
        pretrained_embeddings: Pre-trained embeddings tensor
        model_save_path: Directory to save model checkpoints
        max_len: Maximum sequence length
        max_norm: Maximum gradient norm for clipping
        adversarial_training: Whether to use adversarial training
        epsilon: Epsilon for adversarial example generation
    """
    # Build vocabulary from training data only
    word2idx, idx2word = build_vocab(train_df['text'])

    # Encode all datasets
    X_train = torch.tensor([encode(t, word2idx, max_len) for t in train_df['text']])
    y_train = torch.tensor(train_df['label'].values, dtype=torch.float32)

    X_val = torch.tensor([encode(t, word2idx, max_len) for t in val_df['text']])
    y_val = torch.tensor(val_df['label'].values, dtype=torch.float32)

    # Initialize BiLSTM model
    model = BiLSTMSpam(vocab_size=len(word2idx), embedding_dim=embedding_dim,
                       pretrained_embeddings=pretrained_embeddings)

    # Training parameters
    num_epochs = 40
    learning_rate = 8e-4
    batch_size = 32

    # Move model to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Create datasets and dataloaders
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # Train BiLSTM with specialized training loop
    best_val_loss = float('inf')

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
