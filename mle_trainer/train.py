from google.cloud import storage
import pandas as pd
import math
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from hypertune import HyperTune
import argparse
import os

# ==========================
# ==== Define Variables ====
# ==========================
# When dealing with a large dataset, it is practical to randomly sample
# a smaller proportion of the data to reduce the time and money cost per iteration.
#
# You should start with a low proportion, and increase it
# after you are able to better estimate how long the training time will be
# under different settings, using different proportions of the training set.
# When you are testing, start with 0.2. You need to change it to 1.0 when you make submissions.
# TODO: Set it to 1.0 when you make submissions
SAMPLE_PROB = 1.0  # Sample 20% of the whole dataset
random.seed(15619)  # Set the random seed to get deterministic sampling results
# update the value using the ID of the GS bucket, WITHOUT "gs://"
# for example, if the GS path of the bucket is gs://my-bucket
# the OUTPUT_BUCKET_ID will be "my-bucket"
OUTPUT_BUCKET_ID = 'ml-fare-prediction-222512-p4ml'
# DO NOT change it
DATA_BUCKET_ID = 'p42ml'
# DO NOT change it
TRAIN_FILE = 'data/cc_nyc_fare_train.csv'


# =========================
# ==== Utility Methods ====
# =========================
def haversine_distance(origin, destination):
    """
    Calculate the spherical distance from coordinates

    :param origin: tuple (lat, lng)
    :param destination: tuple (lat, lng)
    :return: Distance in km
    """
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


def calculateDistance(raw_df, lat1, lon1, lat2, lon2, new_column_name):
    new_column=[]
    for index,row in raw_df.iterrows():
        origin = (row[lat1],row[lon1])
        dest = (row[lat2],row[lon2])
        value = haversine_distance(origin, dest)
        new_column.append(value)
    raw_df.insert(1,new_column_name,new_column)
    return raw_df
    
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
        df[longtitude] = df[longtitude].map(lambda x: (x + 180)/360)
    for latitude in ['pickup_latitude','dropoff_latitude','jfk_latitude','lga_latitude','ewr_latitude',
                     'nyc_latitude']:
        df[latitude] = df[latitude].map(lambda x: x/180)
    
    return df

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

def qcutNumerical(df):  
    return pd.get_dummies(df,prefix = ['outlier'], columns= ['outlier']) 

def parseDatetime(raw_df):
    raw_df.loc[:,'year'] = raw_df.pickup_datetime.apply(lambda t: t.year)
    raw_df.loc[:,'month'] = raw_df.pickup_datetime.apply(lambda t: t.month)
    raw_df.loc[:,'weekday'] = raw_df.pickup_datetime.apply(lambda t: t.weekday())
    raw_df.loc[:,'hour'] = raw_df.pickup_datetime.apply(lambda t: t.hour)
    
#     # one hot encoding
#     categorical_cols = ['year','month','weekday','hour']
    
#     for categorical_col in categorical_cols:
#         raw_df[categorical_col] = raw_df[categorical_col].astype('str')
        
#     raw_df = pd.get_dummies(raw_df,prefix=categorical_cols,columns=categorical_cols)
    
    return raw_df
    
# =====================================
# ==== Define data transformations ====
# =====================================
def process_train_data(raw_df):
    
    # delete some abnormal rows
    raw_df = raw_df[(raw_df['fare_amount'] > 0) & (raw_df['fare_amount'] <= 100)] 

    # frist step to deal with abnormal data point
    cleaned_raw_data = dealWithOutliers(raw_df)
    
    # add airports and convert coodinates to distances
    cleaned_data_with_distances = addThreeAirports(cleaned_raw_data)
    
    # date time one-hot encoding
    cleaned_data_with_distance_and_time = parseDatetime(cleaned_data_with_distances)
    
#     # qcut the numerical variables
#     cleaned_data_with_distance_and_time_qcut = qcutNumerical(cleaned_data_with_distance_and_time)
    
    # return the result
    return cleaned_data_with_distance_and_time


def process_test_data(raw_df):

    # frist step to deal with abnormal data point
    cleaned_raw_data = dealWithOutliers(raw_df)
    
    # add airports and convert coodinates to distances
    cleaned_data_with_distances = addThreeAirports(cleaned_raw_data)
    
    # parse datatime this can be ver different
    cleaned_data_with_distance_and_time = parseDatetime(cleaned_data_with_distances)
    
#     # do the qcut for the numerical variables
#     cleaned_data_with_distance_and_time_qcut = qcutNumerical(cleaned_data_with_distance_and_time)

    return cleaned_data_with_distance_and_time


if __name__ == '__main__':
    # ===========================================
    # ==== Download data from Google Storage ====
    # ===========================================
    print('Downloading data from google storage')
    print('Sampling {} of the full dataset'.format(SAMPLE_PROB))
    input_bucket = storage.Client().bucket(DATA_BUCKET_ID)
    output_bucket = storage.Client().bucket(OUTPUT_BUCKET_ID)
    input_bucket.blob(TRAIN_FILE).download_to_filename('train.csv')

    raw_train = pd.read_csv('train.csv', parse_dates=["pickup_datetime"],
                            skiprows=lambda i: i > 0 and random.random() > SAMPLE_PROB)

    print('Read data: {}'.format(raw_train.shape))

    # =============================
    # ==== Data Transformation ====
    # =============================
    df_train = process_train_data(raw_train)

    # Prepare feature matrix X and labels Y
    X = df_train.drop(['key', 'fare_amount', 'pickup_datetime'], axis=1)
    Y = df_train['fare_amount']
    X_train, X_eval, y_train, y_eval = train_test_split(X, Y, test_size=0.33)
    print('Shape of feature matrix: {}'.format(X_train.shape))

    # ======================================================================
    # ==== Improve model performance with hyperparameter tuning ============
    # ======================================================================
    # You are provided with the code that creates an argparse.ArgumentParser
    # to parse the command line arguments and pass these parameters to ML Engine to
    # be tuned by HyperTune enabled.
    # 
    # You need to update both the code below and config.yaml.

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--job-dir',  # MLE passes this in by default
        required=True
    )

    # the 5 lines of code below parse the --max_depth option from the command line
    # and will convert the value into "args.max_depth"
    # "args.max_depth" will be passed to XGBoost training through the `params` variables
    # i.e., xgb.train(params, ...)
    #
    # the 5 lines match the following YAML entry in `config.yaml`:
    # - parameterName: max_depth
    #   type: INTEGER
    #   minValue: 4
    #   maxValue: 10
    # "- parameterName: max_depth" matches "--max_depth"
    # "type: INTEGER" matches "type=int""
    # "minValue: 4" and "maxValue: 10" match "default=6"
    parser.add_argument(
        '--max_depth',
        default=6,
        type=int
    )

    parser.add_argument(
        '--learning_rate',
        default=0.1,
        type=float
    )

    parser.add_argument(
        '--n_estimators',
        default=100,
        type=int
    )

    parser.add_argument(
        '--subsample',
        default=1,
        type=float
    )

    args = parser.parse_args()
    params = {
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': args.subsample,
        'n_estimators': args.n_estimators
    }

    """
    DO NOT CHANGE THE CODE BELOW
    """
    # ===============================================
    # ==== Evaluate performance against test set ====
    # ===============================================
    # Create DMatrix for XGBoost from DataFrames
    d_matrix_train = xgb.DMatrix(X_train, y_train)
    d_matrix_eval = xgb.DMatrix(X_eval)
    model = xgb.train(params, d_matrix_train)
    y_pred = model.predict(d_matrix_eval)
    rmse = math.sqrt(mean_squared_error(y_eval, y_pred))
    print('RMSE: {:.3f}'.format(rmse))

    # Return the score back to HyperTune to inform the next iteration
    # of hyperparameter search
    hpt = HyperTune()
    hpt.report_hyperparameter_tuning_metric(
        hyperparameter_metric_tag='nyc_fare',
        metric_value=rmse)

    # ============================================
    # ==== Upload the model to Google Storage ====
    # ============================================
    JOB_NAME = os.environ['CLOUD_ML_JOB_ID']
    TRIAL_ID = os.environ['CLOUD_ML_TRIAL_ID']
    model_name = 'model.bst'
    model.save_model(model_name)
    blob = output_bucket.blob('{}/{}_rmse{:.3f}_{}'.format(
        JOB_NAME,
        TRIAL_ID,
        rmse,
        model_name))
    blob.upload_from_filename(model_name)
