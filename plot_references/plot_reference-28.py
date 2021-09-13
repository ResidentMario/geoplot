import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd

boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
injurious_collisions = gpd.read_file(gplt.datasets.get_path('nyc_injurious_collisions'))

ax = gplt.voronoi(injurious_collisions.head(1000))
gplt.polyplot(boroughs, ax=ax)