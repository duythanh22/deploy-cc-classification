import torch
import PIL
import os
import logging
import sys
import time
import threading
import psutil

import litserve as ls
from litserve.utils import PickleableHTTPException
from litserve.middlewares import MaxSizeMiddleware

from fastapi import Depends, HTTPException, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from torchvision import transforms
from prometheus_client import (
    CollectorRegistry,
    Counter,
    Histogram,
    Gauge,
    make_asgi_app,
    multiprocess
)

from loguru import logger
import torch.cuda as cuda

# Import your configuration (assumed to exist)
from config.config import config


class PrometheusMonitor(ls.Logger):
    def __init__(self):
        super().__init__()
        # Performance Metrics
        self.requests_total = Counter(
            "api_requests_total",
            "Total number of API requests",
            ["endpoint", "method", "status_code"],
            registry=registry
        )
        self.request_duration = Histogram(
            "api_request_duration_seconds",
            "API request latency",
            ["endpoint", "method"],
            registry=registry
        )
        self.active_requests = Gauge(
            "api_active_requests",
            "Number of currently processing requests",
            registry=registry
        )

        # Model Metrics
        self.model_predictions = Counter(
            "model_predictions_total",
            "Total number of model predictions",
            ["predicted_class", "confidence_level"],
            registry=registry
        )

        # Error Metrics
        self.errors_total = Counter(
            "api_errors_total",
            "Total number of API errors",
            ["error_type", "endpoint"],
            registry=registry
        )

        # Resource Metrics
        self.cpu_usage = Gauge(
            "process_cpu_usage_percent",
            "CPU usage of the API process",
            registry=registry
        )
        self.memory_usage = Gauge(
            "process_memory_usage_bytes",
            "Memory usage of the API process",
            registry=registry
        )
        self.file_descriptors = Gauge(
            "process_open_file_descriptors",
            "Number of open file descriptors",
            registry=registry
        )
        self.thread_count = Gauge(
            "process_threads",
            "Number of threads in the process",
            registry=registry
        )

        # Model Memory Metrics
        self.model_memory_usage = Gauge(
            "model_memory_usage_bytes",
            "Memory used by the model",
            ["device_type"],
            registry=registry
        )

    def monitor_system_resources(self):
        """Periodic system resource monitoring"""
        try:
            process = psutil.Process()

            # CPU Usage
            self.cpu_usage.set(process.cpu_percent())

            # Memory Usage
            self.memory_usage.set(process.memory_info().rss)

            # File Descriptors
            self.file_descriptors.set(process.num_fds())

            # Thread Count
            self.thread_count.set(process.num_threads())

            # GPU Memory if available
            if cuda.is_available():
                allocated = cuda.memory_allocated() / (1024 ** 3)  # GB
                peak = cuda.max_memory_allocated() / (1024 ** 3)  # GB
                self.model_memory_usage.labels(device_type="gpu_allocated").set(allocated * (1024 ** 3))
                self.model_memory_usage.labels(device_type="gpu_peak").set(peak * (1024 ** 3))

        except Exception as e:
            logger.error(f"Resource monitoring error: {e}")

    def process(self, key, value):
        # Request start/end tracking
        if key == "request_start":
            self.active_requests.inc()
        elif key == "request_end":
            self.active_requests.dec()

        # Model prediction metrics
        if key == "model_prediction":
            predicted_class, confidence = value
            self.model_predictions.labels(
                predicted_class=predicted_class,
                confidence_level=self._confidence_bucket(confidence)
            ).inc()

    def _confidence_bucket(self, confidence):
        """Convert confidence to buckets"""
        if confidence < 0.5:
            return "low"
        elif confidence < 0.8:
            return "medium"
        else:
            return "high"

    def log_error(self, error_type, endpoint):
        """Log API errors"""
        self.errors_total.labels(
            error_type=error_type,
            endpoint=endpoint
        ).inc()


# Set up Prometheus multiprocess registry
os.environ["PROMETHEUS_MULTIPROC_DIR"] = "/tmp/prometheus_multiproc_dir"
if not os.path.exists("/tmp/prometheus_multiproc_dir"):
    os.makedirs("/tmp/prometheus_multiproc_dir")

registry = CollectorRegistry()
multiprocess.MultiProcessCollector(registry)


class CervicalCellClassifierAPI(ls.LitAPI):
    security = HTTPBearer()

    def setup(self, devices):
        self.device = devices[0] if isinstance(devices, list) else devices
        try:
            self.model = torch.load(config.MODEL_PATH, map_location=self.device)
            self.model.eval()
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            raise PickleableHTTPException(status_code=500, detail=f"Model load error: {e}")

    def decode_request(self, request: UploadFile, **kwargs) -> torch.Tensor:
        try:
            # Validate and process image
            image = PIL.Image.open(request.file)
            file_ext = os.path.splitext(request.filename)[1].lower()

            if file_ext not in ['.png', '.jpg', '.jpeg']:
                raise PickleableHTTPException(
                    status_code=400,
                    detail=f"Unsupported image format: {file_ext}"
                )

            image = image.convert("RGB")
            image = image.resize(config.IMAGE_INPUT_SIZE, PIL.Image.NEAREST)

            tensor_image = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.IMAGE_MEAN,
                    std=config.IMAGE_STD
                )
            ])(image)

            return tensor_image.unsqueeze(0).to(self.device, non_blocking=True)

        except Exception as e:
            logger.error(f"Image processing error: {e}")
            raise PickleableHTTPException(status_code=500, detail="Image processing failed")

    def predict(self, x: torch.Tensor, **kwargs):
        try:
            # Log request start
            if hasattr(self, "log"):
                self.log("request_start", 1)
                start_time = time.time()

            with torch.inference_mode():
                output = self.model(x)
                probabilities = torch.nn.functional.softmax(output, dim=1)
                _, predicted = torch.max(output.data, 1)

            # Log model prediction
            predicted_class = config.CLASS_NAMES[predicted.item()]
            confidence = probabilities[0][predicted.item()].item()

            if hasattr(self, "log"):
                self.log("model_prediction", (predicted_class, confidence))
                end_time = time.time()
                self.log("inference_time", end_time - start_time)

            return predicted, probabilities

        except Exception as e:
            # Log error metrics
            if hasattr(self, "log"):
                self.log_error(type(e).__name__, "predict")
            raise
        finally:
            # Always log request end
            if hasattr(self, "log"):
                self.log("request_end", 1)

    def encode_response(self, output, **kwargs):
        predicted_label, probabilities = output

        def process_prediction(label, probs):
            label_idx = label.item()
            predicted_class = config.CLASS_NAMES[label_idx]
            class_probabilities = {
                config.CLASS_NAMES[i]: prob.item()
                for i, prob in enumerate(probs)
            }
            confidence_score = probs[label_idx].item()

            return {
                "predicted_class": predicted_class,
                "class_probabilities": class_probabilities,
                "confidence_score": confidence_score
            }

        # Handle batch or single prediction
        if predicted_label.dim() > 0:
            return [process_prediction(label, probs) for label, probs in zip(predicted_label, probabilities)]
        else:
            return process_prediction(predicted_label, probabilities)

    def authorize(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        if token != config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid token")


def periodic_resource_monitor(monitor):
    """Background thread for periodic resource monitoring"""
    while True:
        monitor.monitor_system_resources()
        time.sleep(15)  # Check every 15 seconds


def configure_logging():
    """Configure logging with Loguru"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    logger.add(
        "logs/api.log",
        rotation="100 MB",
        retention="1 week",
        level="DEBUG"
    )


def main():
    # Configure logging
    configure_logging()

    # Create Prometheus monitor
    prometheus_monitor = PrometheusMonitor()

    # Start resource monitoring in background
    resource_thread = threading.Thread(
        target=periodic_resource_monitor,
        args=(prometheus_monitor,),
        daemon=True
    )
    resource_thread.start()

    try:
        # Initialize API and server
        api = CervicalCellClassifierAPI()
        server = ls.LitServer(api, loggers=prometheus_monitor)

        # Mount Prometheus metrics endpoint
        server.app.mount("/api/v1/metrics", make_asgi_app(registry=registry))

        # Add middleware
        server.app.add_middleware(MaxSizeMiddleware, max_size=config.MAX_IMAGE_SIZE)

        logger.info(f"Starting server on port {config.SERVER_PORT}")
        server.run(port=config.SERVER_PORT)

    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()