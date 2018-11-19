import googlemaps
import os


class GoogleMapsClient:
    def __init__(self):
        self.client = googlemaps.Client(key=os.environ['GOOGLE_MAPS_API_KEY'])

    def directions(self, origin, destination):
        directions_result = self.client.directions(origin,
                                                   destination,
                                                   mode="driving")

        return directions_result
