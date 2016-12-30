import sys; sys.path.insert(0, '../')
import geoplot.crs as gcrs
import geoplot as gplt
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import matplotlib.pyplot as plt


# Shape the data.
listings = pd.read_csv("../../data/boston_airbnb/listings.csv")
boston_zip_codes = gpd.read_file("../../data/boston_airbnb/ZIPCODES_NT_POLY.shp")
airbnbs = listings[['latitude', 'longitude']].apply(lambda srs: Point(srs['longitude'], srs['latitude']),
                                                    axis='columns')
listings = gpd.GeoDataFrame(data=listings, geometry=airbnbs)
listings['price'] = listings['price'].map(
    lambda p: p[1:].replace(".", "").replace(",", "") if (not isinstance(p, float)) else np.nan).astype(float)
listings['price'] = list(map(lambda p: float(p) / 100, listings['price']))
listings = listings[listings['price'].notnull()]
listings['zipcode'] = listings['zipcode'].map(lambda z: float(str(z).replace("-", "").split(" ")[0]))
listings = listings[listings['zipcode'].notnull()]
boston_zip_codes['POSTCODE'] = boston_zip_codes['POSTCODE'].map(lambda p: float(p[1:]) if p[0] == '0' else float(p))
boston_zip_codes = boston_zip_codes.drop_duplicates('POSTCODE')
boston_zip_codes = boston_zip_codes.set_index("POSTCODE")
listings = listings[~listings['zipcode'].isin([2218.0, 21341704.0])]
boston_zip_codes = boston_zip_codes.to_crs(epsg=4326)


# Plot the data.
ax = gplt.polyplot(boston_zip_codes.geometry, projection=gcrs.AlbersEqualArea(),
                   facecolor='lightgray', edgecolor='gray', linewidth=0)
gplt.aggplot(listings, projection=gcrs.AlbersEqualArea(), hue='price',
             by='zipcode', geometry=boston_zip_codes.geometry, agg=np.median, ax=ax,
             linewidth=0)


ax.set_title("Median AirBnB Price by Boston Zip Code, 2016")
plt.savefig("boston-airbnb-aggplot.png", bbox_inches='tight', pad_inches=0.1)
