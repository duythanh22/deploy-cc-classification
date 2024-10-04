# Copyright The Lightning AI team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import requests
import json
from config.config import config

# url
# API_URL = "https://8000-01j90j16a5cshabpyej4snevya.cloudspaces.litng.ai/predict"
API_URL = config.API_URL

def send_request(image_path):
    """
    Sends an image to the API server for classification.

    Parameters
    ----------
    image_path : str
        The file path of the image to be sent to the server.

    Returns
    -------
    dict
        The server response containing the classification result.
    """
    headers = {
        "Authorization": f"Bearer {config.API_AUTH_TOKEN}"
    }
    with open(image_path, "rb") as image_file:
        files = {'request': image_file}
        response = requests.post(API_URL, headers=headers, files=files, verify=False)

    return response.json()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Send image to Cervical Cell Classifier API")
    parser.add_argument("image_path", type=str, help="Path to the image file to classify")
    args = parser.parse_args()

    result = send_request(args.image_path)

    print(f"Predicted Class: {result['predicted_class']}")
    print(f"Class Probabilities: {result['class_probabilities']}")


