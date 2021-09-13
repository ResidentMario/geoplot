import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd

contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))

gplt.cartogram(contiguous_usa, scale='population', projection=gcrs.AlbersEqualArea())