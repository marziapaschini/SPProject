import folium
import pandas as pd

from folium.plugins import MarkerCluster
from geopy.geocoders import Nominatim

df = pd.read_csv('weather.csv')
city_names = df["Location"].unique()

mappa = folium.Map(location=[45.523064, -122.676483], zoom_start=4)

marker_cluster = MarkerCluster().add_to(mappa)
geolocator = Nominatim(user_agent="my_app_name")
for city in city_names:
    location = geolocator.geocode(city)
    if location is not None:
        folium.Marker(location=[location.latitude, location.longitude]).add_to(marker_cluster)

mappa