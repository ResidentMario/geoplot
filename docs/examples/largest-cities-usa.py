import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

# load the data
continental_usa_cities = gpd.read_file(gplt.datasets.get_path('usa_cities'))
continental_usa_cities = continental_usa_cities.query('STATE not in ["AK", "HI", "PR"]')
contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))

poly_kwargs = {'linewidth': 0.5, 'edgecolor': 'gray'}
point_kwargs = {'linewidth': 0.5, 'edgecolor': 'black', 'alpha': 1}
legend_kwargs = {'bbox_to_anchor': (0.9, 0.9), 'frameon': False}

proj = gcrs.AlbersEqualArea(central_longitude=-98, central_latitude=39.5)
ax = gplt.polyplot(contiguous_usa, projection=proj, **poly_kwargs)
gplt.pointplot(
    continental_usa_cities,
    scale='POP_2010', limits=(1, 80),
    hue='POP_2010', cmap='Blues',
    legend=True, legend_var='scale',
    legend_values=[8000000, 6000000, 4000000, 2000000, 100000],
    legend_labels=['8 million', '6 million', '4 million', '2 million', '100 thousand'],
    legend_kwargs=legend_kwargs,
    **point_kwargs,
    ax=ax
)

plt.title("Large cities in the contiguous United States, 2010")
plt.savefig("largest-cities-usa.png", bbox_inches='tight', pad_inches=0.1)