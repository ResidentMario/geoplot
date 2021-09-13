import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd

cities = gpd.read_file(gplt.datasets.get_path('usa_cities'))
gplt.pointplot(cities)