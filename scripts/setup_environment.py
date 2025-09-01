"""
Environment setup script for HR Performance Analytics Pro
"""
import os
import sys
import subprocess
from pathlib import Path
from loguru import logger

def create_directories():
    """Create necessary directories"""
    directories = [
        "logs",
        "models",
        "data/raw",
        "data/processed",
        "data/exports",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def setup_environment():
    """Setup development environment"""
    logger.info("Setting up HR Performance Analytics Pro environment...")
    
    # Create directories
    create_directories()
    
    # Create .env file if it doesn't exist
    if not os.path.exists(".env"):
        logger.info("Creating .env file from template...")
        with open(".env.example", "r") as template:
            content = template.read()
        
        with open(".env", "w") as env_file:
            env_file.write(content)
        
        logger.info("Created .env file - please update with your credentials")
    
    # Install dependencies
    logger.info("Installing Python dependencies...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        logger.info("Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install dependencies: {e}")
        return False
    
    logger.info("Environment setup completed successfully!")
    return True

if __name__ == "__main__":
    setup_environment()
