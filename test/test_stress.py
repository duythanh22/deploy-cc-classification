import argparse
import time
import asyncio
import aiohttp
import json
import os
from config.config import config

API_URL = config.API_URL
IMAGE_DIR = "../image_test"

async def send_request(session, image_path):
    headers = {
        "Authorization": f"Bearer {config.API_AUTH_TOKEN}"
    }
    with open(image_path, "rb") as image_file:
        data = aiohttp.FormData()
        data.add_field('request', image_file, filename=os.path.basename(image_path), content_type='image/png')
        start_time = time.time()
        try:
            async with session.post(API_URL, headers=headers, data=data, ssl=False) as response:
                end_time = time.time()
                response_time = end_time - start_time
                if response.status == 200:
                    json_response = await response.json()
                    return json_response, response_time
                else:
                    return {"error": response.status, "detail": await response.text()}, response_time
        except Exception as e:
            return {"error": str(e)}, None

async def send_batch_requests(image_paths):
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, path) for path in image_paths]
        return await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stress test for Cervical Cell Classifier API")
    parser.add_argument("--image_dir", type=str, default=IMAGE_DIR, help="Directory containing the images")
    parser.add_argument("--initial-requests", type=int, default=10, help="Initial number of requests to send")
    parser.add_argument("--max-requests", type=int, default=100, help="Maximum number of requests to send in one cycle")
    parser.add_argument("--step", type=int, default=10, help="Step increase in the number of requests")
    parser.add_argument("--cycles", type=int, default=5, help="Number of cycles to repeat the stress test")
    args = parser.parse_args()

    # Load all images from the directory
    image_paths = [os.path.join(args.image_dir, img) for img in os.listdir(args.image_dir) if img.lower().endswith(('png', 'jpg', 'jpeg'))]

    # Ensure we have enough images for the stress test, repeat images if necessary
    if len(image_paths) < args.max_requests:
        image_paths *= (args.max_requests // len(image_paths)) + 1

    loop = asyncio.get_event_loop()

    # Mở tệp để ghi kết quả
    with open("../test_results.txt", "w") as results_file:
        for cycle in range(args.cycles):
            num_requests = args.initial_requests + cycle * args.step
            if num_requests > args.max_requests:
                num_requests = args.max_requests

            batch_image_paths = image_paths[:num_requests]

            print(f"\nCycle {cycle + 1}/{args.cycles}: Sending {len(batch_image_paths)} requests to {API_URL}...")
            start_time = time.time()
            results = loop.run_until_complete(send_batch_requests(batch_image_paths))
            end_time = time.time()

            total_time = end_time - start_time
            successful_requests = 0
            failed_requests = 0
            total_response_time = 0

            for i, (result, response_time) in enumerate(results):
                if "error" not in result:
                    successful_requests += 1
                    total_response_time += response_time
                else:
                    failed_requests += 1
                print(f"Result for request {i + 1}:")
                print(json.dumps(result, indent=2))
                print(f"Response time: {response_time:.2f} seconds" if response_time else "No response time recorded")
                print()

                # Ghi kết quả vào tệp
                results_file.write(f"Result for request {i + 1}:\n")
                results_file.write(json.dumps(result, indent=2) + "\n")
                results_file.write(
                    f"Response time: {response_time:.2f} seconds\n" if response_time else "No response time recorded\n")
                results_file.write("\n")

            # Ghi tóm tắt cho chu kỳ vào tệp
            results_file.write(f"\nSummary for cycle {cycle + 1}:\n")
            results_file.write(f"Total requests sent: {len(batch_image_paths)}\n")
            results_file.write(f"Successful requests: {successful_requests}\n")
            results_file.write(f"Failed requests: {failed_requests}\n")
            if successful_requests > 0:
                results_file.write(f"Average response time: {total_response_time / successful_requests:.2f} seconds\n")
            results_file.write(f"Cycle duration: {total_time:.2f} seconds\n")

        print(f"\nStress test completed. Total cycles: {args.cycles}")
        results_file.write(f"\nStress test completed. Total cycles: {args.cycles}\n")
    print(f"\nStress test completed. Total cycles: {args.cycles}")
