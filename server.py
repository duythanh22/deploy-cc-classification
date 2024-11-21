import torch
import PIL
import os
import logging
from loguru import logger
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
import sys
from prometheus_client import CollectorRegistry, Histogram, make_asgi_app, multiprocess
import litserve
import time

# Set the directory for multiprocess mode
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc_dir"

# Ensure the directory exists
if not os.path.exists("/tmp/prometheus_multiproc_dir"):
    os.makedirs("/tmp/prometheus_multiproc_dir")

# Use a multiprocess registry
registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)

# Define transformations and class names
class_name = config.CLASS_NAMES
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
])


class PrometheusLogger(litserve.Logger):
    def __init__(self):
        super().__init__()
        self.function_duration = Histogram("request_processing_seconds", "Time spent processing request",
                                           ["function_name"], registry=registry)

    def process(self, key, value):
        print("processing", key, value)
        self.function_duration.labels(function_name=key).observe(value)

class CervicalCellClassifierAPI(ls.LitAPI):
    security = HTTPBearer()

    def setup(self, devices):
        self.device = devices[0] if isinstance(devices, list) else devices
        try:
            self.model = torch.load(config.MODEL_PATH, map_location=self.device)
            logger.info("Setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise PickleableHTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        self.model.eval()

    def decode_request(self, request: UploadFile, **kwargs) -> torch.Tensor:
        try:
            image = PIL.Image.open(request.file)

            file_ext = os.path.splitext(request.filename)[1].lower()
            if file_ext not in ['.png', '.jpg', '.jpeg']:
                raise PickleableHTTPException(status_code=400,
                                              detail=f"Unsupported image format: {file_ext}")

            image = image.convert("RGB")

            image = image.resize(config.IMAGE_INPUT_SIZE, PIL.Image.NEAREST)

            tensor_image = transforms.ToTensor()(image)
            tensor_image = transforms.Normalize(
                mean=config.IMAGE_MEAN,
                std=config.IMAGE_STD
            )(tensor_image)

            return tensor_image.unsqueeze(0).to(self.device, non_blocking=True)

        except PickleableHTTPException:
            raise
        except Exception as e:
            logger.error(f"Error in decode_request: {e}")
            raise PickleableHTTPException(status_code=500, detail="Image processing failed")

    def batch(self, inputs: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(inputs, dim=0).to(self.device)

    def predict(self, x: torch.Tensor, **kwargs):
        try:
            logger.info("Starting prediction...")
            start_time = time.time()

            # Dự đoán
            with torch.inference_mode():
                output = self.model(x)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)

            # Đo thời gian hoàn thành
            end_time = time.time()
            time_taken = end_time - start_time

            # Ghi nhận mức sử dụng bộ nhớ
            try:
                allocated, peak = self.engine.get_memory_usage() if hasattr(self, "engine") else (None, None)
            except Exception as mem_err:
                logger.warning(f"Failed to retrieve memory usage: {str(mem_err)}")
                allocated, peak = None, None

            # Log thông tin
            logger.info(f"Prediction completed in {time_taken:.4f} seconds.")
            if allocated is not None and peak is not None:
                logger.info(f"Memory usage - Allocated: {allocated:.2f} GB, Peak: {peak:.2f} GB")
            logger.debug(f"Predicted labels: {predicted.tolist()}")
            logger.debug(f"Probabilities: {probabilities.tolist()}")

            # Ghi số liệu vào Prometheus (nếu có Histogram)
            if hasattr(self, "log"):
                self.log("inference_time", time_taken)

            return predicted, probabilities

        except Exception as e:
            logger.error(f"Error during prediction: {str(e)}")
            raise PickleableHTTPException(status_code=500, detail="Prediction failed")

    def unbatch(self, output):
        predicted, probabilities = output
        return list(zip(predicted, probabilities))

    def encode_response(self, output, **kwargs):
        predicted_label, probabilities = output

        def process_prediction(label, probs):
            label_idx = label.item()
            if 0 <= label_idx < len(class_name):
                predicted_class = class_name[label_idx]
                class_probabilities = {class_name[i]: prob.item() for i, prob in enumerate(probs)}
                confidence_score = probs[label_idx].item()

                return {
                    "predicted_class": predicted_class,
                    "class_probabilities": class_probabilities,
                    "confidence_score": confidence_score
                }

            logger.error(f"Invalid prediction index: {label_idx}")
            raise PickleableHTTPException(status_code=500, detail="Invalid model prediction")

        if predicted_label.dim() > 0:
            return [process_prediction(label, probs) for label, probs in zip(predicted_label, probabilities)]
        else:
            return process_prediction(predicted_label, probabilities)

    def authorize(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        if token != config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")


if __name__ == '__main__':
    prometheus_logger = PrometheusLogger()
    # prometheus_logger.mount(path="/api/v1/metrics", app=make_asgi_app(registry=registry))
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{"
               "function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/api.log",
        rotation="100 MB",
        retention="1 week",
        level="DEBUG"
    )
    try:
        api = CervicalCellClassifierAPI()
        server = ls.LitServer(api)

        server.app.mount("/api/v1/metrics", make_asgi_app(registry=registry))

        logger.info("Starting server on port 8000")
        server.run(port=config.SERVER_PORT)

    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)
