import os
import torch
import litserve as ls
from PIL import Image
from typing import List
from torchvision import transforms
from fastapi import Depends, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from concurrent.futures import ThreadPoolExecutor
from config.config import config

class_name = config.CLASS_NAMES
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
])

class CervicalCellClassifierAPI(ls.LitAPI):
    security = HTTPBearer()

    def __init__(self):
        super().__init__()
        self.model = None
        self.pool = None
        self.precision = torch.float32
        self.device = None

    def setup(self, devices):
        self.device = devices[0] if isinstance(devices, list) else devices
        self.model = torch.load(config.MODEL_PATH, map_location=self.device)
        self.model.eval()
        self.pool = ThreadPoolExecutor(max_workers=max(os.cpu_count(), 8))

    def decode_request(self, request: List[UploadFile], **kwargs) -> torch.Tensor:
        def process_image(upload_file: UploadFile):
            pil_image = Image.open(upload_file.file).convert("RGB")
            processed_image = transform(pil_image)
            return processed_image

        processed_images = [process_image(file) for file in request]
        return torch.stack(processed_images).to(self.device)

    def batch(self, image_data_list: List[UploadFile]) -> torch.Tensor:
        def process_image(upload_file: UploadFile):
            pil_image = Image.open(upload_file.file).convert('RGB')
            processed_image = transform(pil_image)
            return processed_image

        inputs = list(self.pool.map(process_image, image_data_list))
        return torch.stack(inputs).to(self.device).to(self.precision)

    def predict(self, x: torch.Tensor, **kwargs):
        with torch.no_grad():
            output = self.model(x)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, dim=1)

        predicted_label = class_name[predicted.item()]
        class_probabilities = {class_name[i]: prob.item() for i, prob in enumerate(probabilities[0])}
        return predicted_label, class_probabilities

    def encode_response(self, output, **kwargs):
        predicted_label, class_probabilities = output
        return {
            "predicted_label": predicted_label,
            "class_probabilities": class_probabilities
        }

    def authorize(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        if token != config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")


if __name__ == '__main__':
    api = CervicalCellClassifierAPI()
    server = ls.LitServer(api)
    server.run(port=config.SERVER_PORT)
