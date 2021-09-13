import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd

la_flights = gpd.read_file(gplt.datasets.get_path('la_flights'))
world = gpd.read_file(gplt.datasets.get_path('world'))

ax = gplt.sankey(la_flights, projection=gcrs.Mollweide())
gplt.polyplot(world, ax=ax, facecolor='lightgray', edgecolor='white')
ax.set_global(); ax.outline_patch.set_visible(True)