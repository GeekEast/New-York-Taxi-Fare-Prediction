import json
import logging
import os
import base64
import pandas as pd
import math
import requests
import numpy as np
import pandas as pd
from flask import Flask, request, redirect, url_for, jsonify
from clients.ml_engine import MLEngineClient
from clients.speech_to_text import SpeechToTextClient
from clients.text_to_speech import TextToSpeechClient
from clients.natural_language import NaturalLanguageClient
from clients.google_maps import GoogleMapsClient
from clients.cloud_vision import CloudVisionClient
from clients.automl_client import AutoMLClient

app = Flask(__name__)

project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
mle_model_name = os.getenv("GCP_MLE_MODEL_NAME")
mle_model_version = os.getenv("GCP_MLE_MODEL_VERSION")

ml_engine_client = MLEngineClient(project_id, mle_model_name, mle_model_version)

# =========================
# ==== Utility Methods ====
# =========================
# formula to calculate distance between two points on the map
def haversine_distance(origin, destination):
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = radius * c

    return d

# calculate the distance between two coordinates
def calculateDistance(raw_df, lat1, lon1, lat2, lon2, new_column_name):
    new_column=[]
    for index,row in raw_df.iterrows():
        origin = (row[lat1],row[lon1])
        dest = (row[lat2],row[lon2])
        value = haversine_distance(origin, dest)
        new_column.append(value)
    raw_df.insert(1,new_column_name,new_column)
    return raw_df
    
# add three big airports and the distances
def addThreeAirports(df):
    
    df.loc[:,'jfk_latitude'] = 40.6413111
    df.loc[:,'jfk_longtitude'] = -73.7781391
    
    df.loc[:,'lga_latitude'] = 40.7730136
    df.loc[:,'lga_longtitude'] = -73.8702300
    
    df.loc[:,'ewr_latitude'] = 40.7026322
    df.loc[:,'ewr_longtitude'] = -74.1877659
    
    df.loc[:,'nyc_longtitude'] = -73.9776225
    df.loc[:,'nyc_latitude'] = 40.7638356
    
    df = addDistances(df)
    
    # normalize the lontitude and latitude
    for longtitude in ['pickup_longitude','dropoff_longitude','jfk_longtitude','lga_longtitude',
                      'ewr_longtitude','nyc_longtitude']:
        df[longtitude] = df[longtitude].map(lambda x: (x+180)/360)
    for latitude in ['pickup_latitude','dropoff_latitude','jfk_latitude','lga_latitude','ewr_latitude',
                     'nyc_latitude']:
        df[latitude] = df[latitude].map(lambda x: x/180)
    
    return df

# add distances
def addDistances(df):
    # calculate the trip distance
    calculateDistance(df,'pickup_latitude','pickup_longitude','dropoff_latitude','dropoff_longitude','trip_distance')
    
    # calcualte the distance between pick up geo point and airports
    for end in ['pickup','dropoff']:
        for airport in ['jfk','lga','ewr','nyc']:
                calculateDistance(df, end + '_latitude',end + '_longitude',
                                  airport + '_latitude',airport + '_longtitude',
                                  end + '_to_' + airport)
    return df

# deal with outliers
def dealWithOutliers(train_data):
    lat_upper = 41.0
    lat_lower = 40.0
    long_upper = -73.5
    long_lower = -74.5
    
    train_data['outlier'] = 1
    train_data.loc[train_data.isnull().any(axis=1), 'outlier'] = 0
    for longtitude in ['pickup_longitude','dropoff_longitude']:
        train_data.loc[train_data[longtitude] > long_upper,'outlier'] = 0
        train_data.loc[train_data[longtitude] < long_lower,'outlier'] = 0
        train_data.loc[train_data[longtitude] == 0.0,'outlier'] = 0
    for latitude in ['pickup_latitude','dropoff_latitude']:
        train_data.loc[train_data[latitude] > lat_upper,'outlier'] = 0
        train_data.loc[train_data[latitude] < lat_lower,'outlier'] = 0
        train_data.loc[train_data[longtitude] == 0.0,'outlier'] = 0
    
    return train_data

# deal with datetime
def parseDatetime(raw_df):
    raw_df.loc[:,'year'] = raw_df.pickup_datetime.apply(lambda t: t.year)
    raw_df.loc[:,'month'] = raw_df.pickup_datetime.apply(lambda t: t.month)
    raw_df.loc[:,'weekday'] = raw_df.pickup_datetime.apply(lambda t: t.weekday())
    raw_df.loc[:,'hour'] = raw_df.pickup_datetime.apply(lambda t: t.hour)
    
    return raw_df


# =====================================
# ==== Define data transformations ====
# =====================================
def process_test_data(raw_df):

    # frist step to deal with abnormal data point
    cleaned_raw_data = dealWithOutliers(raw_df)
    
    # add airports and convert coodinates to distances
    cleaned_data_with_distances = addThreeAirports(cleaned_raw_data)
    
    # parse datatime this can be ver different
    cleaned_data_with_distance_and_time = parseDatetime(cleaned_data_with_distances)

    # delete the pickup_datetime, we have to delete the pickup datetime here
    candidate = cleaned_data_with_distance_and_time.drop(['pickup_datetime'],axis=1)
    
    return candidate


@app.route('/')
def index():
    return "Hello"


@app.route('/predict', methods=['POST'])
def predict():
    raw_data_df = pd.read_json(request.data.decode('utf-8'),
                               convert_dates=["pickup_datetime"])
    predictors_df = process_test_data(raw_data_df)
    return json.dumps(ml_engine_client.predict(predictors_df.values.tolist()))

def predict_internal():
    raw_data_df =  pd.read_csv('./temp.csv',parse_dates=["pickup_datetime"])
    raw_data_df.to_csv("test.csv",index=False)
    predictors_df = process_test_data(raw_data_df)
    predicted_fare = ml_engine_client.predict(predictors_df.values.tolist())[0]
    return predicted_fare

@app.route('/farePrediction', methods=['POST'])
def fare_prediction():
    
    # get the speech data
    speech_data = request.data
    
    # transform the speech data into text
    text_from_speech = speech2text(speech_data)
    
    # recognize the places from the text
    places = namedEntities(text_from_speech)
    
    # get the coordinates of the places
    coordinates = get_directions(places[0],places[1])
    
    # set the pickup point and the drop off point
    pickup = coordinates['start_location']
    dropoff = coordinates['end_location']
    
    # prepare the data for the prediction
    pickup_datetime = "2010-05-21 06:28:12 UTC"
    passenger_count = "3"
    pickup_latitude = str(pickup['lat'])
    pickup_longitude = str(pickup['lng'])
    dropoff_longitude = str(dropoff['lng'])
    dropoff_latitude = str(dropoff['lat'])
    
    with open("./temp.csv","w") as f:
        f.write("pickup_datetime,pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,passenger_count\n")
        f.write("{},{},{},{},{},{}\n".format(pickup_datetime,
                                             pickup_latitude, pickup_longitude,
                                             dropoff_latitude, dropoff_longitude,
                                             passenger_count))
    
    fare_amount_float = predict_internal()
    text = "Your expected fare from {} to {} is ${:.2f}".format(places[0], places[1], fare_amount_float*0.5)
    speech = text2speech(text)
    
    result = {"predicted_fare": "{:.2f}".format(fare_amount_float*0.5),
             "entities": places,
             "text":text,
             "speech":speech
             }
    
    return json.dumps(result)


@app.route('/speechToText', methods=['POST'])
def speech_to_text():
    content = base64.b64decode(request.data)
    speech_to_text_client = SpeechToTextClient()
    response = speech_to_text_client.recognize(content)
    return json.dumps({"text":response})

def speech2text(data):
    content = base64.b64decode(data)
    speech_to_text_client = SpeechToTextClient()
    response = speech_to_text_client.recognize(content)
    return response

@app.route('/textToSpeech', methods=['GET'])
def text_to_speech():
    text = request.args.get("text")
    text_to_speech_client = TextToSpeechClient()
    response = text_to_speech_client.synthesize_speech(text)
    audio = str(base64.b64encode(response).decode("utf-8"))
    return json.dumps({"speech":audio})

def text2speech(text):
    text_to_speech_client = TextToSpeechClient()
    response = text_to_speech_client.synthesize_speech(text)
    audio = str(base64.b64encode(response).decode("utf-8"))
    return audio

@app.route('/farePredictionVision', methods=['POST'])
def fare_prediction_vision():
    vision_client = CloudVisionClient()
    automl_client = AutoMLClient()
  
    source = base64.b64decode(request.form.get('source'))
    dest = base64.b64decode(request.form.get('destination'))
    
    # use the vision api at first
    source_text = vision_client.get_landmarks(source)
    dest_text = vision_client.get_landmarks(dest)
    
    # check the automl api then
    if (source_text is None):
        source_text = automl_client.automl_predict(source)
    if (dest_text is None):
        dest_text = automl_client.automl_predict(dest)

    places = [source_text,dest_text]
    print(places)
    # double check if they are none or not
    if ((source_text is not None) and (dest_text is not None)):
        coordinates = get_directions(places[0], places[1])
        # set the pickup point and the drop off point
        pickup = coordinates['start_location']
        dropoff = coordinates['end_location']
    
        # prepare the data for the prediction
        pickup_datetime = "2013-05-21 06:28:12 UTC"
        passenger_count = "1"
        pickup_latitude = str(pickup['lat'])
        pickup_longitude = str(pickup['lng'])
        dropoff_longitude = str(dropoff['lng'])
        dropoff_latitude = str(dropoff['lat'])
    
        with open("./temp2.csv","w") as f:
            f.write("pickup_datetime,pickup_latitude,pickup_longitude,dropoff_latitude,dropoff_longitude,passenger_count\n")
            f.write("{},{},{},{},{},{}\n".format(pickup_datetime,
                                                 pickup_latitude, pickup_longitude,
                                                 dropoff_latitude, dropoff_longitude,
                                                 passenger_count))
    
        fare_amount_float = predict_internal()
        text = "Your expected fare from {} to {} is ${:.2f}".format(places[0], places[1], fare_amount_float*0.5)
        speech = text2speech(text)
    
        result = {"predicted_fare": "{:.2f}".format(fare_amount_float*0.5),
                 "entities": places,
                 "text":text,
                 "speech":speech
                 }
        return json.dumps(result)
    return json.dumps({"result":"unknown"})


@app.route('/namedEntities', methods=['GET'])
def named_entities():
    text = request.args.get("text")
    nl_client = NaturalLanguageClient()
    response = nl_client.analyze_entities(text)
    entities = [i.name for i in response]
    return json.dumps({"entities":entities}) # str(entities)

def namedEntities(text):
    nl_client = NaturalLanguageClient()
    response = nl_client.analyze_entities(text)
    entities = [i.name for i in response]
    return entities
    

@app.route('/directions', methods=['GET'])
def directions():
    origin = request.args.get("origin")
    dest = request.args.get("destination")
    google_map_client = GoogleMapsClient()
    response = google_map_client.directions(origin,dest)
    parent = response[0]['legs'][0]
    res = {"start_location":parent['start_location'],"end_location":parent['end_location']}
    return json.dumps(res)

def get_directions(origin,dest):
    google_map_client = GoogleMapsClient()
    response = google_map_client.directions(origin, dest)
    parent = response[0]['legs'][0]
    res = {"start_location":parent['start_location'],"end_location":parent['end_location']}
    return res
    
@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return """
    An internal error occurred: <pre>{}</pre>
    See logs for full stacktrace.
    """.format(e), 500

if __name__ == '__main__':
    app.debug=True
    app.run()
