"""
Quadtree of NYC traffic collisions
==================================

This example plots traffic collisions in New York City. Overlaying a ``pointplot`` on a
``quadtree`` like this communicates information on two visual channels, position and texture,
simultaneously.
"""


import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

nyc_boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
collisions = gpd.read_file(gplt.datasets.get_path('nyc_collision_factors'))

ax = gplt.quadtree(
    collisions, nmax=1,
    projection=gcrs.AlbersEqualArea(), clip=nyc_boroughs,
    facecolor='lightgray', edgecolor='white', zorder=0
)
gplt.pointplot(collisions, s=1, ax=ax)

plt.title("New York Ciy Traffic Collisions, 2016")
plt.savefig("nyc-collisions-quadtree.png", bbox_inches='tight', pad_inches=0)
