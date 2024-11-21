import argparse
import time
import asyncio
import aiohttp
import json
import os
from config.config import config

API_URL = config.API_URL
IMAGE_DIR = "../image_test"
TIMEOUT_THRESHOLD = 10  # Thời gian timeout tính bằng giây

async def send_request(session, image_path, request_id):
    headers = {
        "Authorization": f"Bearer {config.API_AUTH_TOKEN}"
    }
    with open(image_path, "rb") as image_file:
        data = aiohttp.FormData()
        data.add_field('request', image_file, filename=os.path.basename(image_path), content_type='image/png')
        start_time = time.time()
        try:
            async with session.post(API_URL, headers=headers, data=data, ssl=False, timeout=TIMEOUT_THRESHOLD) as response:
                end_time = time.time()
                response_time = end_time - start_time
                if response.status == 200:
                    json_response = await response.json()
                    return {"id": request_id, "result": json_response, "response_time": response_time}
                else:
                    return {"id": request_id, "error": response.status, "detail": await response.text(), "response_time": response_time}
        except asyncio.TimeoutError:
            return {"id": request_id, "error": "Request timed out", "response_time": None}
        except Exception as e:
            return {"id": request_id, "error": str(e), "response_time": None}

async def send_batch_requests(image_paths):
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, path, request_id=i+1) for i, path in enumerate(image_paths)]
        return await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test Cervical Cell Classifier API")
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR, help="Directory containing the images")
    parser.add_argument("--num-requests", type=int, default=10, help="Number of requests to send")
    args = parser.parse_args()

    # Load all images from the directory
    image_paths = [os.path.join(args.image_dir, img) for img in os.listdir(args.image_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Ensure we have enough images for the load tests, repeat images if necessary
    if len(image_paths) < args.num_requests:
        image_paths *= (args.num_requests // len(image_paths)) + 1

    # Trim to the exact number of requests needed
    image_paths = image_paths[:args.num_requests]

    loop = asyncio.get_event_loop()

    # Start the load tests
    print(f"Sending {len(image_paths)} requests to {API_URL}...")
    start_time = time.time()
    results = loop.run_until_complete(send_batch_requests(image_paths))
    end_time = time.time()

    total_time = end_time - start_time
    successful_requests = 0
    failed_requests = 0
    timeout_requests = 0
    total_response_time = 0

    first_timeout_request = None  # Để lưu request đầu tiên bị timeout

    for result in results:
        request_id = result["id"]
        if "error" not in result:
            successful_requests += 1
            total_response_time += result["response_time"]
        else:
            failed_requests += 1
            if result["error"] == "Request timed out":
                timeout_requests += 1
                if first_timeout_request is None:
                    first_timeout_request = request_id  # Ghi nhận request đầu tiên bị timeout

        print(f"Result for request {request_id}:")
        print(json.dumps(result, indent=2))
        print(f"Response time: {result['response_time']:.2f} seconds" if result["response_time"] else "No response time recorded")
        print()

    # Print summary
    print(f"Total requests sent: {len(image_paths)}")
    print(f"Successful requests: {successful_requests}")
    print(f"Failed requests: {failed_requests}")
    print(f"Timed out requests: {timeout_requests}")
    if successful_requests > 0:
        print(f"Average response time: {total_response_time / successful_requests:.2f} seconds")
    print(f"Total test duration: {total_time:.2f} seconds")

    if first_timeout_request:
        print(f"First request that timed out: Request {first_timeout_request}")
    else:
        print("No requests timed out.")
