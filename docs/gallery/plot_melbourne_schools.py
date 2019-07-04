"""
Voronoi of Melbourne primary schools
====================================

This example shows a ``pointplot`` combined with a ``voronoi`` mapping primary schools in
Melbourne. Schools in outlying, less densely populated areas serve larger zones than those in
central Melbourne.

This example inspired by the `Melbourne Schools Zones Webmap <http://melbourneschoolzones.com/>`_.
"""

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import pandas as pd
import matplotlib.pyplot as plt

melbourne = gpd.read_file(gplt.datasets.get_path('melbourne'))
melbourne_primary_schools = gpd.read_file(gplt.datasets.get_path('melbourne_schools'))\
    .query('School_Type == "Primary"')


ax = gplt.voronoi(
    melbourne_primary_schools, clip=melbourne, linewidth=0.5, edgecolor='white',
    projection=gcrs.Mercator()
)
gplt.polyplot(melbourne, edgecolor='None', facecolor='lightgray', ax=ax)
gplt.pointplot(melbourne_primary_schools, color='black', ax=ax, s=1, extent=melbourne.total_bounds)
plt.title('Primary Schools in Greater Melbourne, 2018')
plt.savefig("melbourne-schools.png", bbox_inches='tight', pad_inches=0)
