import sys; sys.path.insert(0, '../')
import geoplot.crs as ccrs
import geoplot as gplt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt
import shapely


# Load data.
census_tracts = gpd.read_file("../../data/ny_census_2010/cty036.shp")
percent_white = census_tracts['WHITE'] / census_tracts['POP2000']


# Plot data.
gplt.choropleth(census_tracts, hue=percent_white, projection=ccrs.AlbersEqualArea(),
                cmap='Purples', linewidth=0.5, k=None, legend=True)
plt.title("Percentage White Residents, 2000")
plt.savefig("ny-state-demographics.png", bbox_inches='tight', pad_inches=0.1)