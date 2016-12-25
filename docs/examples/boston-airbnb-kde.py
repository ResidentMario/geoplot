import sys; sys.path.insert(0, '../')
import geoplot as gplt
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import shapely
import matplotlib.pyplot as plt
import mplleaflet


# Shape the data.
census_tracts = gpd.read_file("../data/boston_airbnb/Census2010_Tracts.shp")
census_tracts = census_tracts.to_crs(epsg=4326)
listings = pd.read_csv("../data/boston_airbnb/listings.csv")
airbnbs = listings[['latitude', 'longitude']].apply(lambda srs: Point(srs['longitude'], srs['latitude']), axis='columns')
listings = gpd.GeoDataFrame(data=listings, geometry=airbnbs)
listings['price'] = listings['price'].map(lambda p: p[1:].replace(".", "").replace(",", "")).astype(float)
listings = listings[['price', 'geometry']].dropna()
boston_polygon = shapely.ops.cascaded_union(census_tracts.geometry.values)


# Plot the data.

# We're building a webmap, so we'll first create an unprojected map.
ax = gplt.kdeplot(listings)
gplt.polyplot(gpd.GeoSeries([boston_polygon]), edgecolor='white', linewidth=4, ax=ax)

# Now we'll output this map to mplleaflet to generate our webmap. In this example we'll actually go one step further,
# and use a non-default tile layer as well. The default mplleaflet webmap uses the default Leaflet tile service,
# which is Open Street Map (OSM). OSM works great in a lot of cases, but tends to be very busy at a local level (an
# actual strategic choice on the part of the OSM developers, as the higher visibility rewards contributions to the
# project).
#
# Luckily Leaflet (and, by extension, mplleaflet) can be made to work with any valid time service. To do this we can use
# the mplleaflet.fig_to_html method, which creates a string (which we'll write to a file) containing our desired
# data. Here is the method signature that we need:
# >>> mplleaflet.fig_to_html(<matplotlib.Figure>, tiles=(<tile url>, <attribution string>)
# For this demo we'll use the super-basic Hydda.Base tile layer.
#
# For a list of possible valid inputs:
# https://leaflet-extras.github.io/leaflet-providers/preview/
# For the full fig_to_html method signature run mplleaflet.fig_to_html? in IPython or see further:
# https://github.com/jwass/mplleaflet/blob/master/mplleaflet/_display.py#L26
fig = plt.gcf()
with open("boston_kde.html", 'w') as f:
    f.write(
        mplleaflet.fig_to_html(fig, tiles=('http://{s}.tile.openstreetmap.se/hydda/base/{z}/{x}/{y}.png',
                                       'Tiles courtesy of <a href="http://openstreetmap.se/" target="_blank">OpenStreetMap Sweden</a> &mdash; Map data &copy; <a href="http://www.openstreetmap.org/copyright">OpenStreetMap</a>'))
    )