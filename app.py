# app

import torch
from flask import Flask, render_template, request, jsonify
import requests
from typing import Dict, Any
from config.config import config
import urllib3

app = Flask(__name__)
app.secret_key = config.SECRET_KEY
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

API_URL = config.API_URL

def get_prediction(image) -> Dict[str, Any]:
    """
    Send the uploaded image to the API and get the prediction.
    """
    headers = {
        "Authorization": f"Bearer {config.API_AUTH_TOKEN}"
    }
    files = {"request": image}
    try:
        response = requests.post(API_URL, headers=headers, files=files, verify=False, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        # Capture specific error details from the API response
        try:
            error_detail = response.json().get("detail", "Unknown error")
        except (ValueError, AttributeError):
            error_detail = "Failed to get prediction from API"
        app.logger.error(f"API request failed: {str(e)} - Detail: {error_detail}")
        return {"error": error_detail}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "image" not in request.files:
            return jsonify({"error": "No file part"})

        file = request.files["image"]
        if file.filename == "":
            return jsonify({"error": "No selected file"})

        if file:
            # Call the get_prediction function to send the image to the API
            result = get_prediction(file)

            if "error" in result:
                # Return error in JSON response
                return jsonify({"error": result["error"]})

            return jsonify(result)

    return render_template("index.html")

@app.errorhandler(500)
def server_error(error):
    app.logger.error('Server Error: %s', str(error))
    return jsonify({"error": "Internal Server Error"}), 500


if __name__ == "__main__":
    app.run(host=config.FLASK_HOST,
            port=config.FLASK_PORT,
            debug=config.FLASK_DEBUG)
