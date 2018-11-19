"""
A short script to test the local server.
"""
import wave
import requests
import base64
import json
import pandas as pd
from clients.cloud_vision import CloudVisionClient
from clients.automl_client import AutoMLClient
endpoint = "https://ml-fare-prediction-222512.appspot.com"
# endpoint = "http://localhost:5000"

def test_predict():
    data = pd.read_csv('input.csv').to_json(orient='records')
    print(requests.post('{}/predict'.format(endpoint), data=data).json())
    

def test_speech_to_text():
    with open('./1.wav', 'rb') as audio:
        audio_data = base64.encodebytes(audio.read())
        r = requests.post('{}/speechToText'.format(endpoint), data=audio_data)
    print(r.text)
    
def test_text_to_speech():
    text = "I would like to go from Central Park Zoo to Bronx Zoo."
    r = requests.get("{}/textToSpeech".format(endpoint),params={'text': text})
    with wave.open('CentralParkZooToBronxZoo.wav', 'wb') as reply:
        speech_wave_frames = base64.decodebytes(r.json()['speech'].encode('utf-8'))
        
        reply.setnchannels(1)
        reply.setsampwidth(2)
        reply.setframerate(16000)
        reply.writeframes(speech_wave_frames)

def test_named_entities():
    text = "American Museum of Natural History and Bryant Park"
    r = requests.get("{}/namedEntities".format(endpoint),params={'text': text})
    print(r.text)
    
    
def test_directions():
    origin = "Pennsylvania Station"
    dest = "Times Square"
    r = requests.get("{}/directions".format(endpoint),params = {"origin":origin,"destination":dest})
    print(r.text)
    
def test_fare_prediction():
    with open('./1.wav', 'rb') as audio:
        audio_data = base64.encodebytes(audio.read())
        r = requests.post('{}/farePrediction'.format(endpoint), data=audio_data)
    print(r.text)
    
    
def test_fare_vision():
    print("======TEST farePredictionVision======")
    ORI_PATH = "/home/clouduser/ProjectMachineLearning/app-engine-pipeline/clients/pic/bamonte_11.jpg"
    DEST_PATH = "/home/clouduser/ProjectMachineLearning/app-engine-pipeline/clients/pic/Katz_s_32.jpg"

    with open(ORI_PATH, 'rb') as ff:
        ori_data = ff.read()
    with open(DEST_PATH, 'rb') as ff:
        dest_data = ff.read()

    ori_data = str(base64.b64encode(ori_data).decode("utf-8"))
    dest_data = str(base64.b64encode(dest_data).decode("utf-8"))
    data = {'source': ori_data, 'destination': dest_data}
    response = requests.post('{}/farePredictionVision'.format(endpoint), data=data)
    res = json.loads(response.text)
    print(res['entities'])

#     print(requests.post('{}/farePredictionVision'.format(endpoint), data=data).content)

# test_predict() 
# test_speech_to_text()
# test_text_to_speech()
# test_named_entities()
# test_directions()
# test_fare_prediction()
test_fare_vision()

    
    
    

    
    
    
    
    
    
    
    
    
    
    