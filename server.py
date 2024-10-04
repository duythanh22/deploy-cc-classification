import torch
import PIL
import io
import base64
import litserve as ls
from torchvision import transforms
from config.config import config
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import UploadFile
from callbacks import PredictTimeLogger

class_name = config.CLASS_NAMES
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
])

class CervicalCellClassifierAPI(ls.LitAPI):

    # HTTPBearer for authentication
    security = HTTPBearer()

    def setup(self, devices):

        self.device = devices[0] if isinstance(devices, list) else devices
        self.model = torch.load(config.MODEL_PATH, map_location=self.device)
        self.model.eval()

    def decode_request(self, request: UploadFile, **kwargs) -> torch.Tensor:


        pil_image = PIL.Image.open(request.file).convert("RGB")
        processed_image = transform(pil_image)
        return processed_image.unsqueeze(0).to(self.device)

    def predict(self, x, **kwargs):

        with torch.inference_mode():
            output = self.model(x)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)

        predicted_label = class_name[predicted.item()]
        class_probabilities = {class_name[i]: prob.item() for i, prob in enumerate(probabilities[0])}

        return predicted_label, class_probabilities

    def encode_response(self, output, **kwargs):

        predicted_label, class_probabilities = output
        return {
            "predicted_class": predicted_label,
            "class_probabilities": class_probabilities
        }

    def authorize(self, credentials: HTTPAuthorizationCredentials = Depends(security)):

        token = credentials.credentials
        if token != config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")


if __name__ == "__main__":
    api = CervicalCellClassifierAPI()
    server = ls.LitServer(api, callbacks=[PredictTimeLogger()])
    server.run(port=config.SERVER_PORT)
