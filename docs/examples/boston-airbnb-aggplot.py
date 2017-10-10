import sys; sys.path.insert(0, '../')
import geoplot.crs as gcrs
import geoplot as gplt
import numpy as np
import matplotlib.pyplot as plt


# Load the data.
boston_zip_codes = gplt.datasets.load('boston-zip-codes')
boston_zip_codes = boston_zip_codes.assign(id=boston_zip_codes.id.astype(float)).set_index('id')

listings = gplt.datasets.load('boston-airbnb-listings')
listings = listings.assign(zipcode=listings.zipcode.astype(float))


# Plot the data.
ax = gplt.polyplot(boston_zip_codes.geometry, projection=gcrs.AlbersEqualArea(),
                   facecolor='lightgray', edgecolor='gray', linewidth=0)

gplt.aggplot(listings, projection=gcrs.AlbersEqualArea(), hue='price',
             by='zipcode', geometry=boston_zip_codes.geometry, agg=np.median, ax=ax,
             linewidth=0)


ax.set_title("Median AirBnB Price by Boston Zip Code, 2016")
plt.savefig("boston-airbnb-aggplot.png", bbox_inches='tight', pad_inches=0.1)
