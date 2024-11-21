# test_batch.py

import argparse
import requests
import json
from config.config import config
import asyncio
import aiohttp


API_URL = config.API_URL

async def send_request(session, image_path):
    headers = {
        "Authorization": f"Bearer {config.API_AUTH_TOKEN}"
    }
    with open(image_path, "rb") as image_file:
        data = aiohttp.FormData()
        data.add_field('request', image_file)
        async with session.post(API_URL, headers=headers, data=data, ssl=False) as response:
            return await response.json()

async def send_batch_requests(image_paths):
    async with aiohttp.ClientSession() as session:
        tasks = [send_request(session, path) for path in image_paths]
        return await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Send images to Cervical Cell Classifier API")
    parser.add_argument("image_paths", type=str, nargs='+', help="Paths to the image files to classify")
    args = parser.parse_args()

    loop = asyncio.get_event_loop()
    results = loop.run_until_complete(send_batch_requests(args.image_paths))

    for i, result in enumerate(results):
        print(f"Result for image {args.image_paths[i]}:")
        print(json.dumps(result, indent=2))
        print()
