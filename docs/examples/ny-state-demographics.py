import sys; sys.path.insert(0, '../')
import geoplot.crs as gcrs
import geoplot as gplt
import geopandas as gpd
import matplotlib.pyplot as plt


# Load the data.
census_tracts = gplt.datasets.load('ny-census-partial')
percent_white = census_tracts['WHITE'] / census_tracts['POP2000']


# Plot the data.
gplt.choropleth(census_tracts, hue=percent_white, projection=gcrs.AlbersEqualArea(),
                cmap='Purples', linewidth=0.5, edgecolor='white', k=None, legend=True)
plt.title("Percentage White Residents, 2000")
plt.savefig("ny-state-demographics.png", bbox_inches='tight', pad_inches=0.1)