import sys; sys.path.insert(0, '../../')
import geoplot as gplt
import geoplot.crs as ccrs
import geopandas as gpd
import matplotlib.pyplot as plt


# Shape the data.
cities = gpd.read_file("../../data/cities/citiesx010g.shp")
continental_cities = cities[cities['STATE'].map(lambda s: s not in ['PR', 'AK', 'HI', 'VI'])]
continental_cities = continental_cities[continental_cities['POP_2010'] > 100000]

usa = gpd.read_file("../../data/united_states/usa.geojson")
continental_usa = usa[~usa['adm1_code'].isin(['USA-3517', 'USA-3563'])]

poly_kwargs = {'linewidth': 0.5, 'edgecolor': 'gray', 'zorder': -1}
point_kwargs = {'linewidth': 0.5, 'edgecolor': 'black', 'alpha': 1}
legend_kwargs = {'bbox_to_anchor': (1.2, 0.9), 'frameon': False}


# Plot the figure.
ax = gplt.polyplot(continental_usa,
                   projection=ccrs.AlbersEqualArea(central_longitude=-98, central_latitude=39.5),
                   **poly_kwargs)
ax.set_ylim((-1597757.3894385984, 1457718.4893930717))
gplt.pointplot(continental_cities, projection=ccrs.AlbersEqualArea(), ax=ax,
               scale='POP_2010', limits=(1, 80),
               hue='POP_2010', cmap='Blues',
               legend=True, legend_var='scale',
               legend_values=[8000000, 6000000, 4000000, 2000000, 100000],
               legend_labels=['8 million', '6 million', '4 million', '2 million', '100 thousand'],
               legend_kwargs=legend_kwargs,
               **point_kwargs)
plt.title("Large cities in the contiguous United States, 2010")
plt.savefig("largest-cities-usa.png", bbox_inches='tight', pad_inches=0.1)