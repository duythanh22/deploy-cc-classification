import torch
import PIL
import os
from loguru import logger
import litserve as ls
from torchvision import transforms
from fastapi import Depends, HTTPException, UploadFile
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
from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request

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
    """
    A logger class that extends `litserve.Logger` to provide Prometheus metrics
    for monitoring API requests, model predictions, and system resource usage.

    Attributes:
        function_duration (Histogram): Tracks the time spent processing requests.
        api_requests (Counter): Counts the total API requests handled.
        api_response_time (Histogram): Measures the response time of API endpoints.
        model_predictions (Counter): Counts the total number of model predictions.
        model_memory_usage (Gauge): Monitors memory usage by the model.
        system_memory_usage (Gauge): Monitors overall system memory usage.
        cpu_usage (Gauge): Tracks CPU usage by the process and system.

    Methods:
        process(key, value): Processes log entries and updates the corresponding
            Prometheus metrics based on the key-value pairs.
        _confidence_bucket(confidence): Converts confidence levels into predefined
            buckets for categorization.
    """

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
                predicted_class, confidence = value
                confidence_bucket = self._confidence_bucket(confidence)

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
            elif key == "api_requests":
                endpoint, method, status = value["endpoint"], value["method"], value["status"]
                self.api_requests.labels(
                    endpoint=endpoint,
                    method=method,
                    status=str(status)
                ).inc(1)

                if "response_time" in value:
                    self.api_response_time.labels(
                        endpoint=endpoint
                    ).observe(value["response_time"])
            else:
                self.function_duration.labels(function_name=key).observe(value)
        except Exception as e:
            logger.error(
                f"PrometheusLogger ran into an error while processing log for key {key} and value {value}: {e}")

    @staticmethod
    def _confidence_bucket(confidence):
        """Convert confidence to buckets"""
        if confidence < 0.5:
            return "low"
        elif confidence < 0.8:
            return "medium"
        else:
            return "high"


class MetricsMiddleware(BaseHTTPMiddleware):
    """
        Middleware for capturing and logging Prometheus metrics related to API requests.

        This middleware intercepts incoming HTTP requests and logs metrics such as
        request endpoint, HTTP method, status code, and response time using a Prometheus
        logger.

        Attributes:
            prometheus_logger: An instance of a logger that processes and records
                Prometheus metrics.

        Methods:
            dispatch(request, call_next):
                Intercepts the request, measures processing time, and logs relevant
                metrics. Passes the request to the next middleware or endpoint handler.
    """

    def __init__(self, app, prometheus_logger):
        super().__init__(app)
        self.prometheus_logger = prometheus_logger

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as e:
            status_code = 500
            raise e
        finally:
            process_time = time.time() - start_time

            # Log the request metrics
            self.prometheus_logger.process("api_requests", {
                "endpoint": request.url.path,
                "method": request.method,
                "status": status_code,
                "response_time": process_time
            })

        return response


class CervicalCellClassifierAPI(ls.LitAPI):
    """
        API class for classifying cervical cell images using a pre-trained model.

        This class extends `ls.LitAPI` and provides methods for setting up the model,
        decoding incoming image requests, batching inputs, making predictions, and
        encoding responses. It also includes authorization and logging of system
        metrics and prediction details.

        Attributes:
            security: An instance of `HTTPBearer` for handling authorization.
            device: The device (CPU or GPU) on which the model is loaded.
            model: The pre-trained model loaded from the specified path in the config.

        Methods:
            setup(devices):
                Loads the model onto the specified device(s) and sets it to evaluation mode.

            decode_request(request, **kwargs) -> torch.Tensor:
                Processes an uploaded image file, converts it to a tensor, and normalizes it.

            batch(inputs: List[torch.Tensor]) -> torch.Tensor:
                Combines a list of input tensors into a single batch tensor.

            predict(x: torch.Tensor, **kwargs):
                Performs prediction on the input tensor, logs metrics, and returns the predicted class and probabilities.

            unbatch(output):
                Splits the batched output into individual predictions and probabilities.

            encode_response(output, **kwargs):
                Formats the prediction output into a JSON-compatible response.

            authorize(credentials: HTTPAuthorizationCredentials = Depends(security)):
                Validates the provided authorization token against the configured API token.
    """
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

    @staticmethod
    def authorize(credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        if token != config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")


if __name__ == '__main__':
    # Clean up old metric files
    multiprocess_dir = "/tmp/prometheus_multiproc_dir"
    if os.path.exists(multiprocess_dir):
        for file in os.listdir(multiprocess_dir):
            os.remove(os.path.join(multiprocess_dir, file))

    prometheus_logger = PrometheusLogger()

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
        server = ls.LitServer(api, api_path="/api/v1/predict", accelerator="auto",
                              max_batch_size=4, timeout=1, track_requests=True,
                              devices="auto", loggers=prometheus_logger)

        # Add the metrics endpoint
        server.app.mount("/api/v1/metrics", make_asgi_app(registry=registry))

        # Add the middlewares
        server.app.add_middleware(MaxSizeMiddleware, max_size=config.MAX_IMAGE_SIZE)
        server.app.add_middleware(MetricsMiddleware, prometheus_logger=prometheus_logger)

        logger.info("Starting server on port 8000")
        server.run(port=config.SERVER_PORT)

    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)


