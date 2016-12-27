import sys; sys.path.insert(0, '../')
import geoplot as gplt
import geoplot.crs as gcrs
import pandas as pd
import matplotlib.pyplot as plt
import cartopy


# This script demonstrates using the cartopy feature interface alongside geoplot.
# For more information visit http://scitools.org.uk/cartopy/docs/latest/matplotlib/feature_interface.html.


# Fetch the data. Note: this script doesn't actually work! For the moment.
airline_city_pairs = pd.read_csv('../../data/world_flights/flights.csv', index_col=0)


# Plot the data.
f, axarr = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={
    'projection': gcrs.Orthographic(central_latitude=40.7128, central_longitude=-74.0059)
})
plt.suptitle('Popular Flights out of Los Angeles, 2016', fontsize=16)
plt.subplots_adjust(top=0.95)

ax = gplt.sankey(airline_city_pairs.query('Origin == "Los Angeles, CA"'), start='Starting Point', end='Ending Point',
                 projection=gcrs.Orthographic(), scale='PASSENGERS', hue='PASSENGERS', cmap='Purples', ax=axarr[0][0])
ax.set_global()
ax.outline_patch.set_visible(True)
ax.coastlines()

ax = gplt.sankey(airline_city_pairs.query('Origin == "Los Angeles, CA"'), start='Starting Point', end='Ending Point',
                 projection=gcrs.Orthographic(), scale='PASSENGERS', hue='PASSENGERS', cmap='Purples', ax=axarr[0][1])
ax.set_global()
ax.outline_patch.set_visible(True)
ax.stock_img()

ax = gplt.sankey(airline_city_pairs.query('Origin == "Los Angeles, CA"'), start='Starting Point', end='Ending Point',
                 projection=gcrs.Orthographic(), scale='PASSENGERS', hue='PASSENGERS', cmap='Purples', ax=axarr[1][0])
ax.set_global()
ax.outline_patch.set_visible(True)
ax.gridlines()
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)

ax = gplt.sankey(airline_city_pairs.query('Origin == "Los Angeles, CA"'), start='Starting Point', end='Ending Point',
                 projection=gcrs.Orthographic(), scale='PASSENGERS', hue='PASSENGERS', cmap='Purples', ax=axarr[1][1])
ax.set_global()
ax.outline_patch.set_visible(True)
ax.coastlines()
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.LAKES)
ax.add_feature(cartopy.feature.RIVERS)

plt.savefig("los-angeles-flights.png", bbox_inches='tight', pad_inches=0.1)