import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

# load the data
collisions = gpd.read_file(gplt.datasets.get_path('nyc_collision_factors'))
boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))

gplt.quadtree(
    collisions, nmax=1,
    projection=gcrs.AlbersEqualArea(), clip=boroughs,
    hue='NUMBER OF PEDESTRIANS INJURED', cmap='Reds', k=None,
    edgecolor='white', legend=True
)

plt.title("Mean Number Pedestrians Injured in Traffic Collisions by Area")
plt.savefig("ny-collision-quadtree.png", bbox_inches='tight', pad_inches=0.1)
