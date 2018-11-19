from google.cloud import texttospeech


class TextToSpeechClient:
    def __init__(self):
        self.client = texttospeech.TextToSpeechClient()

        self.voice_selection_params = texttospeech.types.VoiceSelectionParams(
            language_code='en-US',
            ssml_gender=texttospeech.enums.SsmlVoiceGender.NEUTRAL)

        self.audio_config = texttospeech.types.AudioConfig(
            audio_encoding=texttospeech.enums.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000)

    def synthesize_speech(self, text):
        synthesis_input = texttospeech.types.SynthesisInput(text=text)

        response = self.client.synthesize_speech(synthesis_input, self.voice_selection_params, self.audio_config)

        return response.audio_content
