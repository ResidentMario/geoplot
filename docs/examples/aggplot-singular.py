import sys; sys.path.insert(0, '../')
import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Shape the data.
manhattan = gpd.read_file("../../data/manhattan_mappluto/MNMapPLUTO.shp")
manhattan['YearBuilt'] = manhattan['YearBuilt'].replace(0, np.nan)
manhattan = manhattan[['geometry', 'YearBuilt']].dropna()
manhattan = manhattan.to_crs(epsg=4326)
manhattan = manhattan.reset_index(drop=True)
manhattan = manhattan.reset_index().rename(columns={'index': 'n'})


# Plot the data.

# This plot demonstrates an extremely useful trick. When used with a provided geometry, the aggplot plot type expects
# an iterable of geometries to be used for binning observations. The idea is that, in general, we have n observations
# and some smaller number k of locations containing them, and we will match observations within the same bin,
# average them in some way, and plot the result.
#
# Of course, what if n == k? In other words, what if every observation comes with its own location? In that case we
# can can pass those locations to the ``geometry`` parameter and pass the data's index to the ``by`` parameter,
# and ``aggplot`` will plot all of our records one at a time!
#
# This is a nice feature to have, and very useful for a wide variety of datasets. In this case we are plotting
# building ages in Manhattan using data taken from MapPLUTO
# (http://www1.nyc.gov/site/planning/data-maps/open-data/dwn-pluto-mappluto.page).
#
# Note that this plot is for the purposes of example only: it contains 40,000 geometries (far more than palatable)
# and so takes a long time to render. To explore the data for real take a look at this all-NYC webmap:
# http://pureinformation.net/building-age-nyc/.
ax = gplt.aggplot(manhattan,
                  projection=gcrs.PlateCarree(),
                  geometry=manhattan.geometry,
                  by=pd.Series(manhattan.index),
                  hue='YearBuilt',
                  linewidth=0)


ax.set_title("Buildings in Manhattan by Year Built")
plt.savefig("aggplot-singular.png", bbox_inches='tight', pad_inches=0.1)