"""
KDEPlot of Boston AirBnB Locations
==================================

This example demonstrates a combined application of ``kdeplot`` and ``pointplot`` to a
dataset of AirBnB locations in Boston. The result is outputted to a webmap using the nifty
``mplleaflet`` library. We sample just 1000 points, which captures the overall trend without
overwhelming the renderer.

`Click here to see this plot as an interactive webmap. 
<http://bl.ocks.org/ResidentMario/868ac097d671df1ed5ec83eed048560c>`_
"""

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import mplleaflet

boston_airbnb_listings = gpd.read_file(gplt.datasets.get_path('boston_airbnb_listings'))

ax = gplt.kdeplot(
    boston_airbnb_listings, cmap='viridis', projection=gcrs.WebMercator(), figsize=(12, 12),
    shade=True
)
gplt.pointplot(boston_airbnb_listings, s=1, color='black', ax=ax)
gplt.webmap(boston_airbnb_listings, ax=ax)
plt.title('Boston AirBnB Locations, 2016', fontsize=18)

fig = plt.gcf()
plt.savefig("boston-airbnb-kde.png", bbox_inches='tight', pad_inches=0.1)
# mplleaflet.show(fig)