import os
import sys
import platform

DATA_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))

colab_path = "/content/drive/MyDrive/Projects/spam-detection-data"
windows_path = "D:\\projects\\spam-detection-data"
linux_path = "/home/michael/PycharmProjects/spam-detection-data"

if "google.colab" in sys.modules:
    DATA_PATH = "/content/drive/MyDrive/Projects/spam-detection-data"
elif platform.system() == "Windows" and os.path.exists(windows_path):
    DATA_PATH = windows_path
elif platform.system() == "Linux" and os.path.exists(linux_path):
    DATA_PATH = linux_path

print("DATA_PATH:", DATA_PATH)
