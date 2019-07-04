"""
Sankey of traffic volumes in Washington DC
==========================================

This example demonstrates an application of the ``sankey`` plot type to a geospatial dataset of
`annual average daily traffic volume 
<https://en.wikipedia.org/wiki/Annual_average_daily_traffic>`_ in Washington DC.
"""

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

dc_roads = gpd.read_file(gplt.datasets.get_path('dc_roads'))

gplt.sankey(
    dc_roads, projection=gcrs.AlbersEqualArea(),
    scale='aadt', limits=(0.1, 10), color='black'
)

plt.title("Streets in Washington DC by Average Daily Traffic, 2015")
plt.savefig("dc-street-network.png", bbox_inches='tight', pad_inches=0.1)
