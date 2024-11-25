import os
import sys
from typing import Dict, Any
import requests
import urllib3
from flask import Flask, render_template, request, jsonify
from loguru import logger
from prometheus_client import make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from config.config import config

# Initialize Flask app
app = Flask(__name__)
app.secret_key = config.SECRET_KEY
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Configure logging
logger.remove()
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "logs/flask_app.log",
    rotation="100 MB",
    retention="1 week",
    level="DEBUG"
)


class ImageClassifierClient:
    def __init__(self, api_url: str, auth_token: str):
        self.api_url = api_url
        self.headers = {"Authorization": f"Bearer {auth_token}"}

    def get_prediction(self, image) -> Dict[str, Any]:
        """Send the uploaded image to the API and get the prediction."""
        try:
            image.seek(0)

            # Send file with correct content-type
            files = {
                "request": (
                    image.filename,
                    image.stream,
                    'image/jpeg' if image.filename.lower().endswith(('.jpg', '.jpeg')) else 'image/png'
                )
            }

            response = requests.post(
                self.api_url,
                headers=self.headers,
                files=files,
                verify=False,
                timeout=10
            )
            response.raise_for_status()

            prediction_result = response.json()
            logger.info(f"Prediction received: {prediction_result}")
            return prediction_result

        except requests.RequestException as e:
            try:
                error_detail = response.json().get("detail", "Unknown error")
                logger.error(f"API Response: {response.text}")  # Add log for debug
            except (ValueError, AttributeError, UnboundLocalError):
                error_detail = "Failed to get prediction from API"

            logger.error(f"API request failed: {str(e)} - Detail: {error_detail}")
            return {"error": error_detail}

        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return {"error": str(e)}

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": "An unexpected error occurred"}


# Initialize classifier client
classifier_client = ImageClassifierClient(
    api_url=config.API_URL,
    auth_token=config.API_AUTH_TOKEN
)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            logger.warning("No file part in request")
            return jsonify({"error": "No file part"})

        file = request.files["image"]
        if file.filename == "":
            logger.warning("No selected file")
            return jsonify({"error": "No selected file"})

        # Thêm logging để debug
        logger.debug(f"Received file: {file.filename}")
        logger.debug(f"File content type: {file.content_type}")
        logger.debug(f"File headers: {dict(file.headers)}")

        result = classifier_client.get_prediction(file)
        return jsonify(result)

    return render_template("index.html")


@app.errorhandler(500)
def server_error(error):
    logger.error(f'Server Error: {str(error)}')
    return jsonify({"error": "Internal Server Error"}), 500


# Add prometheus metrics endpoint
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

if __name__ == "__main__":
    try:
        logger.info(f"Starting Flask server on {config.FLASK_HOST}:{config.FLASK_PORT}")
        run_simple(
            hostname=config.FLASK_HOST,
            port=config.FLASK_PORT,
            application=app,
            use_reloader=config.FLASK_DEBUG,
            use_debugger=config.FLASK_DEBUG
        )
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)