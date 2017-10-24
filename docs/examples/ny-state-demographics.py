# Load the data (uses the `quilt` package).
import geopandas as gpd
from quilt.data.ResidentMario import geoplot_data

census_tracts = gpd.read_file(geoplot_data.ny_census_partial())
percent_white = census_tracts['WHITE'] / census_tracts['POP2000']


# Plot the data.
import geoplot.crs as gcrs
import geoplot as gplt
import matplotlib.pyplot as plt

gplt.choropleth(census_tracts, hue=percent_white, projection=gcrs.AlbersEqualArea(),
                cmap='Purples', linewidth=0.5, edgecolor='white', k=None, legend=True)
plt.title("Percentage White Residents, 2000")
plt.savefig("ny-state-demographics.png", bbox_inches='tight', pad_inches=0.1)