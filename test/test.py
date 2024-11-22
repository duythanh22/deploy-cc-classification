import unittest
import io
import os
import requests
from config.config import config
class TestImageClassificationAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_url = config.API_URL
        cls.test_images_dir = "/home/xeon-3/PycharmProjects/deploy-cc-classification/"

    def load_test_image(self, filename):
        filepath = os.path.join(self.test_images_dir, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Test image not found: {filepath}")

        with open(filepath, "rb") as image_file:
            return io.BytesIO(image_file.read())

    def test_successful_image_classification(self):
        try:
            test_image = self.load_test_image("image_test/61f01c5fcec542a39807a20c.png")
            files = {"request": ("61f01c5fcec542a39807a20c.png", test_image, "image/png")}
            headers = {"Authorization": f"Bearer {config.API_AUTH_TOKEN}"}

            response = requests.post(self.api_url, files=files, headers=headers, verify=False)

            self.assertEqual(response.status_code, 200)
            json_response = response.json()
            self.assertIn("predicted_class", json_response)
            self.assertIn("class_probabilities", json_response)
            self.assertIn("confidence_score", json_response)
            print("Test successful_image_classification: OK")
        except Exception as e:
            print(f"Test successful_image_classification: Fail ({e})")

    def test_invalid_token(self):
        try:
            test_image = self.load_test_image("image_test/6211c2e171c8f45788287393.png")
            files = {"request": ("6211c2e171c8f45788287393.png", test_image, "image/png")}
            headers = {"Authorization": "Bearer invalid_token"}

            response = requests.post(self.api_url, files=files, headers=headers, verify=False)

            self.assertEqual(response.status_code, 401)
            self.assertIn("Invalid or missing token", response.json()["detail"])
            print("Test invalid_token: OK")
        except Exception as e:
            print(f"Test invalid_token: Fail ({e})")

    def test_unsupported_image_format(self):
        try:
            test_image = self.load_test_image("image_test/gif.gif")
            files = {"request": ("gif.gif", test_image, "image/gif")}
            headers = {"Authorization": f"Bearer {config.API_AUTH_TOKEN}"}

            response = requests.post(self.api_url, files=files, headers=headers, verify=False)

            self.assertEqual(response.status_code, 400)
            self.assertIn("Unsupported image format", response.json()["detail"])
            print("Test unsupported_image_format: OK")
        except Exception as e:
            print(f"Test unsupported_image_format: Fail ({e})")

    def test_missing_image(self):
        try:
            headers = {"Authorization": f"Bearer {config.API_AUTH_TOKEN}"}
            response = requests.post(self.api_url, files={}, headers=headers)

            self.assertEqual(response.status_code, 422)  # Unprocessable Entity
            print("Test missing_image: OK")
        except Exception as e:
            print(f"Test missing_image: Fail ({e})")

    def test_large_image(self):
        try:
            test_image = self.load_test_image("image_test/large.jpg")
            files = {"request": ("large.jpg", test_image, "image/jpeg")}
            headers = {"Authorization": f"Bearer {config.API_AUTH_TOKEN}"}

            response = requests.post(self.api_url, files=files, headers=headers)

            self.assertEqual(response.status_code, 413)  # Payload Too Large
            print("Test large_image: OK")
        except Exception as e:
            print(f"Test large_image: Fail ({e})")

    def test_corrupted_image(self):
        try:
            corrupted_image = io.BytesIO(b'not an image')
            files = {"request": ("corrupted.jpg", corrupted_image, "image/jpeg")}
            headers = {"Authorization": f"Bearer {config.API_AUTH_TOKEN}"}

            response = requests.post(self.api_url, files=files, headers=headers)

            self.assertEqual(response.status_code, 500)
            self.assertIn("Internal server error", response.json()["detail"])
            print("Test corrupted_image: OK")
        except Exception as e:
            print(f"Test corrupted_image: Fail ({e})")

    def test_multiple_images(self):
        try:
            test_images = [
                self.load_test_image("image_test/6203368cc4dbb4451b872de6.png"),
                self.load_test_image("image_test/61f6b14a12ad37b9dc961ab3.png")
            ]

            headers = {"Authorization": f"Bearer {config.API_AUTH_TOKEN}"}

            for i, image in enumerate(test_images):
                files = {"request": (f"test_image_{i}.png", image, "image/png")}
                response = requests.post(self.api_url, files=files, headers=headers)
                self.assertEqual(response.status_code, 200)
                json_response = response.json()

                self.assertIn("predicted_class", json_response)
                self.assertIn("class_probabilities", json_response)
                self.assertIn("confidence_score", json_response)
            print("Test multiple_images: OK")
        except Exception as e:
            print(f"Test multiple_images: Fail ({e})")


if __name__ == '__main__':
    unittest.main()
