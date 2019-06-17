import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt

# load the data
nyc_boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
nyc_injurious_collisions = gpd.read_file(gplt.datasets.get_path('nyc_injurious_collisions'))

# A plot type using Voronoi tessellation: https://en.wikipedia.org/wiki/Voronoi_diagram

f, axarr = plt.subplots(1, 2, figsize=(16, 8))
axarr[0].axis('off')
axarr[1].axis('off')
gplt.voronoi(
    nyc_injurious_collisions.head(1000),
    edgecolor='lightsteelblue', linewidth=0.5, ax=axarr[0]
)
gplt.polyplot(nyc_boroughs, linewidth=0.5, ax=axarr[0])
gplt.voronoi(
    nyc_injurious_collisions.head(1000),
    hue='NUMBER OF PERSONS INJURED', cmap='Reds',
    edgecolor='white', clip=nyc_boroughs.geometry,
    linewidth=0.5, categorical=True, ax=axarr[1]
)
gplt.polyplot(nyc_boroughs, linewidth=1, ax=axarr[1])
plt.suptitle("Injurious Car Crashes in New York City, 2016", fontsize=20, y=0.95)

plt.savefig("nyc-collisions-voronoi.png", bbox_inches='tight', pad_inches=0)
