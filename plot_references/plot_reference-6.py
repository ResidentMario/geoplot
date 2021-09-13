import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd

boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea())