import sys; sys.path.insert(0, '../')
import geoplot as gplt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt


# Import the data.
census_tracts = gpd.read_file("../../data/boston_airbnb/Census2010_Tracts.shp")
census_tracts = census_tracts.to_crs(epsg=4326)
listings = pd.read_csv("../../data/boston_airbnb/listings.csv")
airbnbs = listings[['latitude', 'longitude']].apply(lambda srs: Point(srs['longitude'], srs['latitude']),
                                                    axis='columns')
listings = gpd.GeoDataFrame(data=listings, geometry=airbnbs)
listings['price'] = listings['price'].map(lambda p: p[1:].replace(".", "").replace(",", "")).astype(float)
listings = listings[['price', 'geometry']].dropna()

#########
# NOTE: In order to improve performance, only a small sample of listings will be used!
#########
listings = listings[['price', 'geometry']].dropna().sample(250)

# Plot the data.
gplt.pointplot(listings, hue='price', cmap='Blues')
fig = plt.gcf()
import mplleaflet
mplleaflet.save_html(fig, fileobj='boston-airbnb-webmap.html')
