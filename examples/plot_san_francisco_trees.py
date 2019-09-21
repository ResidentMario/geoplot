"""
Quadtree of San Francisco street trees
======================================

San Francisco maintains data about city-maintained street trees in the city. Most trees are
identified by their species, but some trees lack this information. This example is a recipe
demonstrating how a ``quadtree`` plot can be used to inspect the geospatial nullity pattern
of a dataset: e.g. whether or not trees in certain areas of the city are less likely to be
classified into a specific species than others.

In this case we see that there is small but significant amount of variation in the percentage
of trees classified per area, which ranges from 88% to 98%.

For more tools for visualizing data nullity, `check out the missingno library
<https://github.com/ResidentMario/missingno>`_.
"""

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt


trees = gpd.read_file(gplt.datasets.get_path('san_francisco_street_trees_sample'))
sf = gpd.read_file(gplt.datasets.get_path('san_francisco'))


ax = gplt.quadtree(
    trees.assign(nullity=trees['Species'].notnull().astype(int)),
    projection=gcrs.AlbersEqualArea(),
    hue='nullity', nmax=1, cmap='Greens', k=5, legend=True,
    clip=sf, edgecolor='white', linewidth=1
)
gplt.polyplot(sf, facecolor='None', edgecolor='gray', linewidth=1, zorder=2, ax=ax)

plt.savefig("san-francisco-street-trees.png", bbox_inches='tight', pad_inches=0)
