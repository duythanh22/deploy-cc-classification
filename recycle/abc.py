# import torch
# import PIL
# import io
# import base64
# import litserve as ls
#
# from torchvision import transforms
# from fastapi import Depends, HTTPException
# from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
# from fastapi import UploadFile
#
# from config.config import config
# from callbacks import PredictTimeLogger
#
#
# class_name = config.CLASS_NAMES
# transform = transforms.Compose([
#     transforms.Resize(config.IMAGE_INPUT_SIZE),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
# ])
#
#
# class CervicalCellClassifierAPI(ls.LitAPI):
#     # HTTPBearer for authentication
#     security = HTTPBearer()
#
#     def setup(self, devices):
#         self.device = devices[0] if isinstance(devices, list) else devices
#         self.model = torch.load(config.MODEL_PATH, map_location=self.device)
#         self.model.eval()
#
#     def decode_request(self, request: UploadFile, **kwargs) -> torch.Tensor:
#         pil_image = PIL.Image.open(request.file).convert("RGB")
#         processed_image = transform(pil_image)
#         return processed_image.unsqueeze(0).to(self.device)
#
#     def predict(self, x, **kwargs):
#         with torch.inference_mode():
#             output = self.model(x)
#             probabilities = torch.nn.functional.softmax(output, dim=1)
#             _, predicted = torch.max(output.data, 1)
#
#         predicted_label = class_name[predicted.item()]
#         class_probabilities = {class_name[i]: prob.item() for i, prob in enumerate(probabilities[0])}
#
#         return predicted_label, class_probabilities
#
#     def encode_response(self, output, **kwargs):
#         predicted_label, class_probabilities = output
#         return {
#             "predicted_class": predicted_label,
#             "class_probabilities": class_probabilities
#         }
#
#     def authorize(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
#         token = credentials.credentials
#         if token != config.API_AUTH_TOKEN:
#             raise HTTPException(status_code=401, detail="Invalid or missing token")
#
#
# if __name__ == "__main__":
#     api = CervicalCellClassifierAPI()
#     server = ls.LitServer(api, callbacks=[PredictTimeLogger()])
#     server.run(port=config.SERVER_PORT)


###################################################################3
import torch
import PIL
import io
import base64
import litserve as ls
import logging

from torchvision import transforms
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi import UploadFile
from litserve.utils import PickleableHTTPException
from litserve.middlewares import MaxSizeMiddleware
from config.config import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class_name = config.CLASS_NAMES
transform = transforms.Compose([
    transforms.Resize(config.IMAGE_INPUT_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=config.IMAGE_MEAN, std=config.IMAGE_STD)
])


class CervicalCellClassifierAPI(ls.LitAPI):
    # HTTPBearer for authentication
    security = HTTPBearer()

    def setup(self, devices):
        self.device = devices[0] if isinstance(devices, list) else devices
        try:
            self.model = torch.load(config.MODEL_PATH, map_location=self.device)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise PickleableHTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
        self.model.eval()

    def decode_request(self, request: UploadFile, **kwargs) -> torch.Tensor:
        try:

            pil_image = PIL.Image.open(request.file)
            image_format = pil_image.format.lower()

            if image_format not in ["png", "jpg", "jpeg"]:
                raise PickleableHTTPException(status_code=400,
                                              detail=f"Unsupported image format: {image_format}. Only PNG and JPG/JPEG are supported.")

            pil_image = pil_image.convert("RGB")
            processed_image = transform(pil_image)
            return processed_image.unsqueeze(0).to(self.device)

        except PickleableHTTPException as pe:
            raise pe
        except Exception as e:
            logger.error(f"Error occurred while processing the image: {str(e)}")
            raise PickleableHTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    def predict(self, x, **kwargs):
        with torch.inference_mode():
            output = self.model(x)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            _, predicted = torch.max(output.data, 1)

            if len(class_name) <= predicted.item():
                logger.error("Model returned an invalid prediction.")
                raise PickleableHTTPException(status_code=500, detail="Model returned an invalid prediction.")

        predicted_label = class_name[predicted.item()]
        class_probabilities = {class_name[i]: prob.item() for i, prob in enumerate(probabilities[0])}

        return predicted_label, class_probabilities

    def encode_response(self, output, **kwargs):
        if not isinstance(output, tuple) or len(output) != 2:
            logger.error("Invalid output format from model.")
            raise PickleableHTTPException(status_code=500, detail="Invalid output format from model.")

        predicted_label, class_probabilities = output
        return {
            "predicted_class": predicted_label,
            "class_probabilities": class_probabilities
        }

    def authorize(self, credentials: HTTPAuthorizationCredentials = Depends(security)):
        token = credentials.credentials
        if token != config.API_AUTH_TOKEN:
            raise HTTPException(status_code=401, detail="Invalid or missing token")


if __name__ == "__main__":
    api = CervicalCellClassifierAPI()
    server = ls.LitServer(api)
    server.app.add_middleware(MaxSizeMiddleware, max_size=config.MAX_IMAGE_SIZE)
    server.run(port=config.SERVER_PORT)
