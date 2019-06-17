import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import numpy as np
import matplotlib.pyplot as plt

# load the data
boston_zip_codes = gpd.read_file(gplt.datasets.get_path('boston_zip_codes'))
boston_zip_codes = boston_zip_codes.assign(id=boston_zip_codes.id.astype(float)).set_index('id')
boston_airbnb_listings = gpd.read_file(gplt.datasets.get_path('boston_airbnb_listings'))

proj = gcrs.AlbersEqualArea()
ax = gplt.polyplot(
    boston_zip_codes, 
    projection=proj,
    facecolor='lightgray',
    edgecolor='gray',
    linewidth=0
)
gplt.aggplot(
    boston_airbnb_listings,
    projection=proj,
    hue='price',
    by='zipcode',
    geometry=boston_zip_codes,
    agg=np.median,
    ax=ax,
    linewidth=0
)

ax.set_title("Median AirBnB Price by Boston Zip Code, 2016")
plt.savefig("boston-airbnb-aggplot.png", bbox_inches='tight', pad_inches=0.1)
