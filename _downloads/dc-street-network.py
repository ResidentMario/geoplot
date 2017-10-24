# Load the data (uses the `quilt` package).
import geopandas as gpd
from quilt.data.ResidentMario import geoplot_data

dc = gpd.read_file(geoplot_data.dc_roads())


# Plot the data.
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

ax = gplt.sankey(dc, path=dc.geometry, projection=gcrs.AlbersEqualArea(), scale='aadt',
                 limits=(0.1, 10))
plt.title("Streets in Washington DC by Average Daily Traffic, 2015")
plt.savefig("dc-street-network.png", bbox_inches='tight', pad_inches=0.1)