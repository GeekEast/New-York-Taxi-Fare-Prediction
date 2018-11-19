from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types


class NaturalLanguageClient:
    def __init__(self):
        self.client = language.LanguageServiceClient()

    def analyze_entities(self, text):
        document = types.Document(
            content=text,
            type=enums.Document.Type.PLAIN_TEXT)

        return self.client.analyze_entities(document).entities
