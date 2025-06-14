import os
import subprocess
import sys
from utils.constants import WORKSPACE_DIR, GIT_REPOSITORY

def run_command(command):
    """Run a shell command and return its output"""
    process = subprocess.Popen(
        command, 
        shell=True, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE
    )
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(stderr.decode())
        return False
    return stdout.decode()

def install_dependencies():
    """Install required Python packages"""
    packages = [
        "transformers==4.48.0"
    ]
    print("Installing dependencies...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + packages)
    
    # Install PyTorch with CUDA support
    torch_command = f"{sys.executable} -m pip install -q torch --index-url https://download.pytorch.org/whl/cu126"
    subprocess.check_call(torch_command, shell=True)
    print("Dependencies installed successfully")

def setup_workspace(branch="master"):
    """Set up the workspace directory and repository"""
    current_dir = os.getcwd()
    
    # Check if we're already in the workspace or if it exists
    if current_dir == WORKSPACE_DIR:
        print(f"Already in workspace directory: {WORKSPACE_DIR}")
        run_command(f"git pull origin {branch}")
        return
    
    # Check if workspace directory exists
    if not os.path.exists(WORKSPACE_DIR):
        print(f"Cloning repository to: {WORKSPACE_DIR}")
        parent_dir = os.path.dirname(WORKSPACE_DIR)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        run_command(f"git clone {GIT_REPOSITORY} {WORKSPACE_DIR}")
    
    # Change to workspace directory
    os.chdir(WORKSPACE_DIR)
    print(f"Changed working directory to: {os.getcwd()}")
    
    # Update repository
    run_command(f"git pull origin {branch}")
    
    # List directory contents
    files = os.listdir(".")
    print(f"Directory contents: {', '.join(files)}")

