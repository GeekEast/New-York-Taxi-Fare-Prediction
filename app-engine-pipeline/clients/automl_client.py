from google.cloud import automl_v1beta1
from google.cloud.automl_v1beta1.proto import service_pb2

class AutoMLClient():
    def __init__(self):
        # create the client
        self.prediction_client = automl_v1beta1.PredictionServiceClient()
        self.label_mappings = {"jing_fong":"Jing Fong",
                               "bamonte":"Bamonte",
                               "katz_deli":"Katz's Delicatessen",
                               "acme":"ACME"}
        
    def automl_predict(self,content):
        
        # define the parameters
        project_id = "ml-fare-prediction-222512" 
        model_id = "ICN2197394516481698826"
        name = 'projects/{}/locations/us-central1/models/{}'.format(project_id, model_id)
        payload = {'image': {'image_bytes': content }}
        params = {}
        
        response = self.prediction_client.predict(name, payload, params)
        display_name = list(response.payload)[0].display_name
        name = None
        
        try:
            name = self.label_mappings[display_name]
        except Exception as e:
            name = display_name
        
        return str(name)
    
    
    

        