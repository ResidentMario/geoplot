"""
Sankey of Los Angeles flight volumes with Cartopy globes
========================================================

This example plots passenger volumes for commercial flights out of Los Angeles International
Airport. Some globe-modification options available in ``cartopy`` are demonstrated. Visit
`the cartopy docs <https://scitools.org.uk/cartopy/docs/latest/matplotlib/feature_interface.html>`_
for more information.
"""

import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import cartopy
import mapclassify as mc

la_flights = gpd.read_file(gplt.datasets.get_path('la_flights'))
scheme = mc.Quantiles(la_flights['Passengers'], k=5)

f, axarr = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={
    'projection': gcrs.Orthographic(central_latitude=40.7128, central_longitude=-74.0059)
})
plt.suptitle('Popular Flights out of Los Angeles, 2016', fontsize=16)
plt.subplots_adjust(top=0.95)

ax = gplt.sankey(
    la_flights, scale='Passengers', hue='Passengers', cmap='Purples', scheme=scheme, ax=axarr[0][0]
)
ax.set_global()
ax.outline_patch.set_visible(True)
ax.coastlines()

ax = gplt.sankey(
    la_flights, scale='Passengers', hue='Passengers', cmap='Purples', scheme=scheme, ax=axarr[0][1]
)
ax.set_global()
ax.outline_patch.set_visible(True)
ax.stock_img()

ax = gplt.sankey(
    la_flights, scale='Passengers', hue='Passengers', cmap='Purples', scheme=scheme, ax=axarr[1][0]
)
ax.set_global()
ax.outline_patch.set_visible(True)
ax.gridlines()
ax.coastlines()
ax.add_feature(cartopy.feature.BORDERS)

ax = gplt.sankey(
    la_flights, scale='Passengers', hue='Passengers', cmap='Purples', scheme=scheme, ax=axarr[1][1]
)
ax.set_global()
ax.outline_patch.set_visible(True)
ax.coastlines()
ax.add_feature(cartopy.feature.LAND)
ax.add_feature(cartopy.feature.OCEAN)
ax.add_feature(cartopy.feature.LAKES)
ax.add_feature(cartopy.feature.RIVERS)

plt.savefig("los-angeles-flights.png", bbox_inches='tight', pad_inches=0.1)
