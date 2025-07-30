import os
import sys
import platform

GIT_REPOSITORY = "https://github.com/RationalEar/spam-detection.git"
IS_COLAB = False
DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "/", "data"))
WORKSPACE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

colab_path = "/content/drive/MyDrive/Projects/spam-detection-data"
windows_path = "D:\\projects\\spam-detection-data"
linux_path = "/home/michael/PycharmProjects/spam-detection-data"

if "google.colab" in sys.modules:
    IS_COLAB = True
    DATA_PATH = "/content/drive/MyDrive/Projects/spam-detection-data"
    WORKSPACE_DIR = "/content/spam-detection"
    
    from google.colab import userdata
    
    os.environ["DATABRICKS_HOST"] = userdata.get('DATABRICKS_HOST')
    os.environ["DATABRICKS_TOKEN"] = userdata.get('DATABRICKS_TOKEN')
    os.environ["MLFLOW_REGISTRY_URI"] = userdata.get('MLFLOW_REGISTRY_URI')
    os.environ["MLFLOW_EXPERIMENT_ID"] = userdata.get('MLFLOW_EXPERIMENT_ID')
    os.environ["MLFLOW_TRACKING_URI"] = userdata.get('MLFLOW_TRACKING_URI')
    
elif platform.system() == "Windows" and os.path.exists(windows_path):
    DATA_PATH = windows_path
elif platform.system() == "Linux" and os.path.exists(linux_path):
    DATA_PATH = linux_path

MODEL_SAVE_PATH = os.path.join(DATA_PATH, "trained-models")
GLOVE_PATH = os.path.join(DATA_PATH, 'data/raw/glove.6B/glove.6B.300d.txt')

os.makedirs(MODEL_SAVE_PATH, exist_ok=True)