# **Spam Detection Project Summary**

---

### **Phase 1: Setup & Data Preparation**
#### **1. Project Setup (Local - PyCharm)**
- **Create Project**:
  - Open PyCharm → New Project → `spam-detection`
  - Set up Python 3.8+ environment (Conda/Virtualenv).

- **File Structure**:
  ```
  spam-detection/
  ├── data/               # Raw/preprocessed data
  ├── models/            # Model definitions (CNN, BiLSTM, etc.)
  ├── utils/             # Helper functions (preprocessing, metrics)
  ├── configs.py         # Hyperparameters
  ├── train.py           # Training script
  └── requirements.txt   # Dependencies
  ```

#### **2. Install Dependencies**
- In PyCharm terminal:
  ```bash
  pip install torch==2.0.1 transformers==4.30.2 scikit-learn pandas numpy matplotlib
  ```
- Save to `requirements.txt`:
  ```bash
  pip freeze > requirements.txt
  ```

#### **3. Download & Preprocess Data**
- **Download SpamAssassin Dataset**:
  ```python
  # utils/data_loader.py
  import pandas as pd
  from sklearn.model_selection import train_test_split

  def load_data():
      # Load emails, labels (adjust paths)
      emails, labels = ...  
      X_train, X_test, y_train, y_test = train_test_split(emails, labels, test_size=0.2)
      return X_train, X_test, y_train, y_test
  ```

---

### **Phase 2: Model Development (Local)**
#### **4. Implement Models**
- **CNN Model** (`models/cnn.py`):
  ```python
  import torch.nn as nn

  class SpamCNN(nn.Module):
      def __init__(self):
          super().__init__()
          self.embedding = nn.Embedding(vocab_size, 300)  # GloVe
          self.conv1 = nn.Conv1d(300, 128, kernel_size=3)
          # ... add layers
      def forward(self, x):
          # Implement forward pass
          return x
  ```

- **BiLSTM Model** (`models/bilstm.py`):
  ```python
  class SpamBiLSTM(nn.Module):
      def __init__(self):
          super().__init__()
          self.lstm = nn.LSTM(input_size=300, hidden_size=128, bidirectional=True)
          # ... add attention
  ```

#### **5. Training Script** (`train.py`)
  ```python
  from models.cnn import SpamCNN
  from utils.data_loader import load_data

  def train():
      X_train, X_test, y_train, y_test = load_data()
      model = SpamCNN()
      # Training loop (optimizer, loss, etc.)
      model.save('cnn_model.pt')
  ```

---

### **Phase 3: Run on Colab**
#### **6. Upload to GitHub**
- Commit code to GitHub:
  ```bash
  git init
  git remote add origin https://github.com/yourusername/spam-detection.git
  git add .
  git commit -m "Initial commit"
  git push -u origin main
  ```

#### **7. Colab Setup**
- Open [Google Colab](https://colab.research.google.com/) → New Notebook.
- Clone repo and install dependencies:
  ```python
  !git clone https://github.com/yourusername/spam-detection.git
  %cd spam-detection
  !pip install -r requirements.txt
  ```

#### **8. Run Training**
- Execute your script:
  ```python
  %run train.py  # Runs on Colab’s GPU
  ```
- For interactive development:
  ```python
  from train import train
  train()  # Call your functions directly
  ```

---

### **Phase 4: Save & Monitor**
#### **9. Save Results**
- Mount Google Drive in Colab:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
- Save models:
  ```python
  torch.save(model, '/content/drive/MyDrive/models/cnn_model.pt')
  ```

#### **10. Monitor Progress**
- Use TensorBoard:
  ```python
  %load_ext tensorboard
  %tensorboard --logdir /content/drive/MyDrive/logs
  ```

---

### **Phase 5: Iterate & Improve**
1. **Test locally** → **Debug in PyCharm** → **Push changes to GitHub**.
2. **Re-run** in Colab:
   ```python
   !git pull origin main  # Sync latest code
   %run train.py
   ```

---

### **Critical Tips**
1. **Colab GPU**: 
   - Runtime → Change runtime type → GPU (T4/A100).
2. **Data Caching**: 
   - Upload preprocessed data to Drive to avoid re-processing.
3. **Session Management**: 
   - Save checkpoints hourly (Colab may disconnect).

---

### **Troubleshooting**
- **OOM Errors**: Reduce batch size (8 or 16).
- **Slow Training**: Use mixed precision (`torch.cuda.amp`).
- **Version Mismatch**: Pin library versions in `requirements.txt`.
