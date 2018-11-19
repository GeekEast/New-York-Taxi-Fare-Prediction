from google.cloud import vision
from google.cloud.vision import types


class CloudVisionClient:
    def __init__(self):
        self.client = vision.ImageAnnotatorClient()

    def get_landmarks(self, content):
        image = types.Image(content=content)

        response = self.client.landmark_detection(image=image)

        for landmark in response.landmark_annotations:
            return landmark.description
