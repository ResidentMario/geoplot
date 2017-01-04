"""
This script tests whether the current environment works correctly or not.
"""

import sys; sys.path.insert(0, '../')
import geoplot as gplt
from geoplot import crs as gcrs
import geopandas as gpd


# cf. https://github.com/Toblerity/Shapely/issues/435

# Fiona/Shapely/Geopandas test.
cities = gpd.read_file("../data/cities/citiesx010g.shp")
census_tracts = gpd.read_file("../data/nyc_census_tracts/census_tracts_2010.geojson")


# Cartopy test.
gplt.pointplot(cities.head(50), extent=(10, 20, 10, 20))