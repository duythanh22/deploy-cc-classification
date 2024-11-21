import psutil
import time
from prometheus_client import Counter, Histogram, Gauge, start_http_server

# Define Prometheus metrics
REQUEST_COUNT = Counter('api_request_count', 'Total number of requests received', ['endpoint'])
REQUEST_SUCCESS_COUNT = Counter('api_request_success_count', 'Number of successful requests', ['endpoint'])
REQUEST_ERROR_COUNT = Counter('api_request_error_count', 'Number of failed requests', ['endpoint', 'error_type'])
REQUEST_IN_PROGRESS = Gauge('api_requests_in_progress', 'Number of requests currently being processed', ['endpoint'])
PREDICTION_TIME = Histogram('api_prediction_time_seconds', 'Time taken for predictions', ['endpoint'])
REQUEST_LATENCY = Histogram(
    'api_request_latency_seconds',
    'Request latency (seconds)',
    ['endpoint'],
    buckets=[0.05, 0.1, 0.25, 0.5, 1, 2.5, 5]
)
SYSTEM_METRICS = Gauge('system_metrics', 'System resource usage', ['resource_type'])
REQUEST_SIZE = Histogram('api_request_size_bytes', 'Size of incoming requests in bytes', ['endpoint'])
RESPONSE_SIZE = Histogram('api_response_size_bytes', 'Size of outgoing responses in bytes', ['endpoint'])
STATUS_CODE_COUNTER = Counter(
    'api_status_code_count',
    'HTTP status codes returned by the API',
    ['endpoint', 'status_code']
)


# Function to start Prometheus HTTP server
def start_monitoring_server(port):
    start_http_server(port)


# Middleware to monitor requests and system metrics
async def monitor_requests(request, call_next):
    endpoint = request.url.path  # Use request path as an endpoint label
    REQUEST_COUNT.labels(endpoint=endpoint).inc()  # Increment request count
    REQUEST_IN_PROGRESS.labels(endpoint=endpoint).inc()  # Increment in-progress requests

    # Handle and observe request size
    content_length = request.headers.get('content-length', 0)
    try:
        content_length = float(content_length)
    except ValueError:
        content_length = 0.0
    REQUEST_SIZE.labels(endpoint=endpoint).observe(content_length)

    # Record CPU, RAM, Disk I/O usage, and system load
    SYSTEM_METRICS.labels(resource_type='cpu_usage').set(psutil.cpu_percent())
    SYSTEM_METRICS.labels(resource_type='ram_usage').set(psutil.virtual_memory().percent)
    SYSTEM_METRICS.labels(resource_type='disk_io_usage').set(
        psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes)
    SYSTEM_METRICS.labels(resource_type='active_connections').set(len(psutil.net_connections(kind='tcp')))
    SYSTEM_METRICS.labels(resource_type='system_load').set(psutil.getloadavg()[0])  # 1-minute load average

    start_time = time.time()
    try:
        # Time taken for the entire request including prediction
        with REQUEST_LATENCY.labels(endpoint=endpoint).time():
            # Track time taken for prediction
            with PREDICTION_TIME.labels(endpoint=endpoint).time():
                response = await call_next(request)

        process_time = time.time() - start_time

        # Track response size
        response_content_length = response.headers.get('content-length', 0)
        try:
            response_content_length = float(response_content_length)
        except ValueError:
            response_content_length = 0.0
        RESPONSE_SIZE.labels(endpoint=endpoint).observe(response_content_length)

        # Track status code
        status_code = response.status_code
        STATUS_CODE_COUNTER.labels(endpoint=endpoint, status_code=status_code).inc()

        # Increment success or error count based on status code
        if 200 <= status_code < 400:
            REQUEST_SUCCESS_COUNT.labels(endpoint=endpoint).inc()
        else:
            error_type = f'HTTP_{status_code}'
            REQUEST_ERROR_COUNT.labels(endpoint=endpoint, error_type=error_type).inc()

        # Add headers with performance and system metrics
        response.headers["X-Process-Time"] = f"{process_time:.4f} seconds"
        response.headers["X-CPU-Usage"] = f"{psutil.cpu_percent()}%"
        response.headers["X-RAM-Usage"] = f"{psutil.virtual_memory().percent}%"
        response.headers[
            "X-Disk-Usage"] = f"{(psutil.disk_io_counters().read_bytes + psutil.disk_io_counters().write_bytes) / (1024 ** 2):.2f} MB"
        response.headers["X-Active-Connections"] = str(len(psutil.net_connections(kind='tcp')))
        response.headers["X-System-Load"] = f"{psutil.getloadavg()[0]:.2f}"

    except Exception as e:
        # Handle unexpected exceptions
        error_type = type(e).__name__
        REQUEST_ERROR_COUNT.labels(endpoint=endpoint, error_type=error_type).inc()
        raise e
    finally:
        REQUEST_IN_PROGRESS.labels(endpoint=endpoint).dec()  # Decrement in-progress requests

    return response