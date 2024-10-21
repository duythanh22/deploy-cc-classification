import torch
import PIL
import os
import logging
import litserve as ls
from torchvision import transforms
from fastapi import Depends, HTTPException, UploadFile, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from litserve.utils import PickleableHTTPException
from litserve.middlewares import MaxSizeMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from config.config import config
from utils.callbacks import PredictTimeLogger
from monitor.monitor import start_monitoring_server, monitor_requests
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define transformations and class names
class_name = config.CLASS_NAMES
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
])


class CervicalCellClassifierAPI(ls.LitAPI):
    security = HTTPBearer()

    def setup(self, devices):
        self.device = devices[0] if isinstance(devices, list) else devices
        try:
            self.model = torch.load(config.MODEL_PATH, map_location=self.device)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise PickleableHTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        self.model.eval()

    def decode_request(self, request: UploadFile, **kwargs) -> torch.Tensor:
        try:
            pil_image = PIL.Image.open(request.file)
            image_format = pil_image.format.lower()

            if image_format not in ["png", "jpg", "jpeg"]:
                raise PickleableHTTPException(status_code=400,
                                              detail=f"Unsupported image format: {image_format}. Only PNG and JPG/JPEG are supported.")

            pil_image = pil_image.convert("RGB")
            processed_image = transform(pil_image)
            return processed_image.unsqueeze(0).to(self.device)
        except PickleableHTTPException as pe:
            raise pe
        except Exception as e:
            logger.error(f"Error occurred while processing the image: {str(e)}")
            raise PickleableHTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def batch(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(inputs, dim=0).to(self.device)

    def predict(self, x: torch.Tensor, **kwargs):
        with torch.inference_mode():
            output = self.model(x)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)

        return predicted, probabilities

    def unbatch(self, output):
        predicted, probabilities = output
        return list(zip(predicted, probabilities))

    def encode_response(self, output, **kwargs):
        predicted_label, probabilities = output
        if isinstance(predicted_label, torch.Tensor) and predicted_label.dim() > 0:
            results = []
            for label, probs in zip(predicted_label, probabilities):
                if label.item() >= len(class_name):
                    logger.error("Model returned an invalid prediction.")
                    raise PickleableHTTPException(status_code=500, detail="Model returned an invalid prediction.")

                predicted_class = class_name[label.item()]
                class_probabilities = {class_name[i]: prob.item() for i, prob in enumerate(probs)}
                confidence_score = probs[label.item()].item()

                results.append({
                    "predicted_class": predicted_class,
                    "class_probabilities": class_probabilities,
                    "confidence_score": confidence_score
                })
            return results
        else:
            if predicted_label.item() >= len(class_name):
                logger.error("Model returned an invalid prediction.")
                raise PickleableHTTPException(status_code=500, detail="Model returned an invalid prediction.")

            predicted_class = class_name[predicted_label.item()]
            class_probabilities = {class_name[i]: prob.item() for i, prob in enumerate(probabilities)}
            confidence_score = probabilities[predicted_label.item()].item()

            return {
                "predicted_class": predicted_class,
                "class_probabilities": class_probabilities,
                "confidence_score": confidence_score
            }

    def authorize(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        if token != config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")


def create_app():
    # Start Prometheus monitoring server
    start_monitoring_server(config.PROMETHEUS_PORT)

    api = CervicalCellClassifierAPI()

    server = ls.LitServer(api, accelerator="cpu", devices=1, timeout=10, max_batch_size=4,
                          batch_timeout=4, callbacks=[PredictTimeLogger()])
    server.app.add_middleware(MaxSizeMiddleware, max_size=config.MAX_IMAGE_SIZE)

    app = server.app

    # Instrument FastAPI app with Prometheus metrics
    Instrumentator().instrument(app).expose(app)

    # Add monitoring middleware
    app.middleware("http")(monitor_requests)

    return app, server


if __name__ == "__main__":
    app, server = create_app()
    server.run(port=config.SERVER_PORT)
