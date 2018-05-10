import quilt
from quilt.data.ResidentMario import geoplot_data
import geopandas as gpd

boroughs = gpd.read_file(geoplot_data.nyc_boroughs())
injurious_collisions = gpd.read_file(geoplot_data.nyc_injurious_collisions())

# A plot type using Voronoi tessellation: https://en.wikipedia.org/wiki/Voronoi_diagram

import geoplot as gplt
import matplotlib.pyplot as plt

f, axarr = plt.subplots(1, 2, figsize=(16, 8))

gplt.voronoi(injurious_collisions.head(1000), edgecolor='lightsteelblue', linewidth=0.5, ax=axarr[0])
gplt.polyplot(boroughs, linewidth=0.5, ax=axarr[0])

gplt.voronoi(injurious_collisions.head(1000), hue='NUMBER OF PERSONS INJURED', cmap='Reds',
             edgecolor='white', clip=boroughs.geometry,
             linewidth=0.5, categorical=True, ax=axarr[1])
gplt.polyplot(boroughs, linewidth=1, ax=axarr[1])

axarr[0].axis('off')
axarr[1].axis('off')

plt.suptitle("Injurious Car Crashes in New York City, 2016", fontsize=20, y=0.95)

plt.savefig("nyc-collisions-voronoi.png", bbox_inches='tight', pad_inches=0)