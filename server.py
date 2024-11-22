import torch
import PIL
import os
from loguru import logger
import litserve as ls
from torchvision import transforms
from fastapi import Depends, HTTPException, UploadFile, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from litserve.utils import PickleableHTTPException
from litserve.middlewares import MaxSizeMiddleware
from config.config import config
from typing import List
import sys
from prometheus_client import CollectorRegistry, Histogram, Gauge, make_asgi_app, multiprocess, Counter
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

        # Timing metrics
        self.function_duration = Histogram(
            "request_processing_seconds",
            "Time spent processing request",
            ["function_name"], registry=registry
        )

        # Request metrics
        self.api_requests = Counter(
            "api_requests_total",
            "Total API requests handled",
            ["endpoint", "method", "status"], registry=registry
        )

        self.api_response_time = Histogram(
            "api_response_time_seconds",
            "Response time of API endpoints",
            ["endpoint"], registry=registry
        )

        # Model prediction metrics
        self.model_predictions = Counter(
            "model_predictions_total",
            "Total number of model predictions",
            ["predicted_class", "confidence_level"], registry=registry
        )

        self.model_memory_usage = Gauge(
            "model_memory_usage_bytes",
            "Memory used by the model",
            ["device_type"], registry=registry
        )

        self.system_memory_usage = Gauge(
            "system_memory_usage_bytes",
            "Overall system memory usage",
            ["memory_type"], registry=registry
        )

        self.cpu_usage = Gauge(
            "cpu_usage_percent",
            "CPU usage by the process and system",
            ["cpu_type"], registry=registry
        )

    def process(self, key, value):
        try:
            # Log processing based on the key
            if key == "model_prediction":
                predicted_class, confidence = value  # Unpack the tuple
                confidence_bucket = self._confidence_bucket(confidence)

                # Increment the Counter by 1
                self.model_predictions.labels(
                    predicted_class=predicted_class,
                    confidence_level=confidence_bucket
                ).inc(1)
            elif key == "model_memory_allocated":
                self.model_memory_usage.labels(device_type="gpu" if cuda.is_available() else "cpu").set(value)
            elif key == "model_memory_peak":
                self.model_memory_usage.labels(device_type="gpu_peak" if cuda.is_available() else "cpu_peak").set(value)
            elif key.startswith("system_memory_"):
                memory_type = key.split("_")[-1]
                self.system_memory_usage.labels(memory_type=memory_type).set(value)
            elif key == "cpu_usage":
                self.cpu_usage.labels(cpu_type="system").set(value["system"])
                self.cpu_usage.labels(cpu_type="process").set(value["process"])
            else:
                self.function_duration.labels(function_name=key).observe(value)
        except Exception as e:
            logger.error(
                f"PrometheusLogger ran into an error while processing log for key {key} and value {value}: {e}")

    def _confidence_bucket(self, confidence):
        """Convert confidence to buckets"""
        if confidence < 0.5:
            return "low"
        elif confidence < 0.8:
            return "medium"
        else:
            return "high"

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

            # Memory and system metrics
            try:
                if cuda.is_available():
                    allocated = cuda.memory_allocated(self.device) / (1024 ** 3)  # Convert to GB
                    peak = cuda.max_memory_allocated(self.device) / (1024 ** 3)  # Convert to GB
                else:
                    allocated = None
                    peak = None

                process = psutil.Process(os.getpid())
                cpu_usage = {
                    "system": psutil.cpu_percent(interval=None),
                    "process": process.cpu_percent(interval=None)
                }

                self.log("cpu_usage", cpu_usage)
                self.log("system_memory_total", psutil.virtual_memory().total)
                self.log("system_memory_available", psutil.virtual_memory().available)
                self.log("system_memory_used", psutil.virtual_memory().used)

                if allocated is not None:
                    self.log("model_memory_allocated", allocated * (1024 ** 3))  # Convert back to bytes
                    self.log("model_memory_peak", peak * (1024 ** 3))

            except Exception as mem_err:
                logger.warning(f"Failed to retrieve system metrics: {mem_err}")

            # Log inference time
            self.log("inference_time", time_taken)

            # Log prediction metrics
            predicted_class = config.CLASS_NAMES[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()
            self.log("model_prediction", (predicted_class, confidence))

            logger.info(f"Prediction completed in {time_taken:.4f} seconds")
            logger.debug(f"Predicted class: {predicted_class}, Confidence: {confidence:.4f}")

            return predicted, probabilities

        except Exception as e:
            logger.error(f"Error during prediction: {e}")
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

        server.app.mount("/api/v2/metrics", make_asgi_app(registry=registry))
        server.app.add_middleware(MaxSizeMiddleware, max_size=config.MAX_IMAGE_SIZE)
        logger.info("Starting server on port 8000")
        server.run(port=config.SERVER_PORT)

    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)