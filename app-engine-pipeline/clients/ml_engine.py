import googleapiclient.discovery


class MLEngineClient:
    """
    You should NOT change this class.
    """
    def __init__(self, project, model, version):
        self.ml_service = googleapiclient.discovery.build('ml', 'v1')
        self.name = 'projects/{}/models/{}/versions/{}'.format(project, model, version)

    def predict(self, instances):
        response = self.ml_service.projects().predict(
            name=self.name,
            body={'instances': instances}
        ).execute()

        if 'error' in response:
            raise RuntimeError(response['error'])

        return response['predictions']
