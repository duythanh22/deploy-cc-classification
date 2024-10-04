import os
import secrets

def generate_secret_key(length=32):
    return secrets.token_hex(length)

class Config:
    # Server Configuration
    SERVER_HOST = '0.0.0.0'
    SERVER_PORT = 8000
    MODEL_PATH = 'model/best_model_checkpoint.pt'
    SERVER_AUTH_TOKEN = os.environ.get('SERVER_AUTH_TOKEN') or generate_secret_key()

    # Flask App Configuration
    FLASK_HOST = '0.0.0.0'
    FLASK_PORT = 5000
    FLASK_DEBUG = False  # Set to False in production
    # API_URL = "https://8000-01j90j16a5cshabpyej4snevya.cloudspaces.litng.ai/predict"
    API_URL = 'http://127.0.0.1:8000/predict'
    SECRET_KEY = os.environ.get('FLASK_SECRET_KEY') or generate_secret_key()

    # Image Processing
    IMAGE_INPUT_SIZE = (224, 224)
    IMAGE_MEAN = [0.485, 0.456, 0.406]
    IMAGE_STD = [0.229, 0.224, 0.225]

    # Class Names
    CLASS_NAMES = ["ASC_H", "ASC_US", "HSIL", "LSIL", "SCC"]

    # Monitoring Dashboard Configuration
    ENABLE_LOGGING = True
    DASHBOARD_CONFIG_FILE = 'dashboard_config.cfg'

    # Security
    API_AUTH_TOKEN = "cc-classify"

    @classmethod
    def print_tokens(cls):
        print(f"SERVER_AUTH_TOKEN: {cls.SERVER_AUTH_TOKEN}")
        print(f"FLASK_SECRET_KEY: {cls.SECRET_KEY}")
        print(f"API_AUTH_TOKEN: {cls.API_AUTH_TOKEN}")


config = Config()

# Uncomment the following line if you want to print the tokens when the config is imported
# config.print_tokens()