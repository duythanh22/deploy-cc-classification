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
from config.config import config
from utils.callbacks import PredictTimeLogger
from monitor.monitor import start_monitoring_server, monitor_requests
from typing import List
import sys
from prometheus_client import CollectorRegistry, Histogram, Gauge, make_asgi_app, multiprocess
import litserve
import time
import psutil
import torch.cuda as cuda

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
        # Existing histogram for request processing time
        self.function_duration = Histogram("request_processing_seconds", "Time spent processing request",
                                           ["function_name"], registry=registry)

        # New Gauges for memory metrics
        self.model_memory_usage = Gauge(
            "model_memory_usage_bytes",
            "Memory used by the model",
            ["device_type"],
            registry=registry
        )

        self.system_memory_usage = Gauge(
            "system_memory_usage_bytes",
            "Overall system memory usage",
            ["memory_type"],
            registry=registry
        )

    def process(self, key, value):
        print("processing", key, value)
        # Existing time processing
        self.function_duration.labels(function_name=key).observe(value)

        # Process memory metrics if they are memory-related
        if key == "model_memory_allocated":
            self.model_memory_usage.labels(device_type="gpu" if cuda.is_available() else "cpu").set(value)
        elif key == "model_memory_peak":
            self.model_memory_usage.labels(device_type="gpu_peak" if cuda.is_available() else "cpu_peak").set(value)
        elif key == "system_memory_total":
            self.system_memory_usage.labels(memory_type="total").set(value)
        elif key == "system_memory_available":
            self.system_memory_usage.labels(memory_type="available").set(value)
        elif key == "system_memory_used":
            self.system_memory_usage.labels(memory_type="used").set(value)


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

            with torch.inference_mode():
                output = self.model(x)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)

            end_time = time.time()
            time_taken = end_time - start_time

            # Memory usage tracking
            try:
                # Model memory usage (GPU or CPU)
                if cuda.is_available():
                    allocated = cuda.memory_allocated(self.device) / (1024 ** 3)  # Convert to GB
                    peak = cuda.max_memory_allocated(self.device) / (1024 ** 3)  # Convert to GB
                else:
                    allocated = None
                    peak = None

                # System memory usage
                process = psutil.Process(os.getpid())
                system_memory_total = psutil.virtual_memory().total
                system_memory_available = psutil.virtual_memory().available
                system_memory_used = psutil.virtual_memory().used
                process_memory = process.memory_info().rss

                # Log memory metrics
                if hasattr(self, "log"):
                    if allocated is not None:
                        self.log("model_memory_allocated",
                                 allocated * (1024 ** 3))  # Convert back to bytes for Prometheus
                        self.log("model_memory_peak", peak * (1024 ** 3))

                    self.log("system_memory_total", system_memory_total)
                    self.log("system_memory_available", system_memory_available)
                    self.log("system_memory_used", system_memory_used)

                # Logging for local debugging
                logger.info(f"Prediction completed in {time_taken:.4f} seconds.")
                if allocated is not None and peak is not None:
                    logger.info(f"Model Memory usage - Allocated: {allocated:.2f} GB, Peak: {peak:.2f} GB")
                logger.info(f"System Memory - Total: {system_memory_total / (1024 ** 3):.2f} GB, "
                            f"Available: {system_memory_available / (1024 ** 3):.2f} GB, "
                            f"Used: {system_memory_used / (1024 ** 3):.2f} GB")
                logger.debug(f"Predicted labels: {predicted.tolist()}")
                logger.debug(f"Probabilities: {probabilities.tolist()}")

                # Log inference time
                if hasattr(self, "log"):
                    self.log("inference_time", time_taken)

            except Exception as mem_err:
                logger.warning(f"Failed to retrieve memory usage: {str(mem_err)}")

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
        server = ls.LitServer(api, loggers=prometheus_logger)

        server.app.mount("/api/v1/metrics", make_asgi_app(registry=registry))
        server.app.add_middleware(MaxSizeMiddleware, max_size=config.MAX_IMAGE_SIZE)
        logger.info("Starting server on port 8000")
        server.run(port=config.SERVER_PORT)

    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)
