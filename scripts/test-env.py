"""
This script tests whether the current environment works correctly or not.
"""

import sys; sys.path.insert(0, '../geoplot/')
import geoplot as gplt
from geoplot import crs as gcrs
import geopandas as gpd


# cf. https://github.com/Toblerity/Shapely/issues/435

# Fiona/Shapely/Geopandas test.
cities = gpd.read_file("../data/cities/citiesx010g.shp")
boroughs = gpd.read_file("../data/nyc_boroughs/boroughs.geojson")


# Cartopy test.
gplt.pointplot(cities.head(50), extent=(10, 20, 10, 20))