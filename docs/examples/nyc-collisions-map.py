import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

# load the data
nyc_boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
nyc_fatal_collisions = gpd.read_file(gplt.datasets.get_path('nyc_fatal_collisions'))
nyc_injurious_collisions = gpd.read_file(gplt.datasets.get_path('nyc_injurious_collisions'))


fig = plt.figure(figsize=(10,5))
proj = projection=gcrs.AlbersEqualArea(central_latitude=40.7128, central_longitude=-74.0059)
ax1 = plt.subplot(121, projection=proj)
ax2 = plt.subplot(122, projection=proj)

gplt.polyplot(nyc_boroughs, ax=ax1, projection=proj)
gplt.pointplot(
    nyc_fatal_collisions, projection=proj,
    hue='BOROUGH', cmap='Set1',
    edgecolor='white', linewidth=0.5,
    scale='NUMBER OF PERSONS KILLED', limits=(2, 8),
    legend=True, legend_var='scale', legend_kwargs={'loc': 'upper left'},
    legend_values=[2, 1], legend_labels=['2 Fatalities', '1 Fatality'],
    ax=ax1
)
ax1.set_title("Fatal Crashes in New York City, 2016")

gplt.polyplot(nyc_boroughs, ax=ax2, projection=proj)
gplt.pointplot(
    nyc_injurious_collisions, projection=proj,
    hue='BOROUGH', cmap='Set1',
    edgecolor='white', linewidth=0.5,
    scale='NUMBER OF PERSONS INJURED', limits=(1, 10),
    legend=True, legend_var='scale', legend_kwargs={'loc': 'upper left'},
    legend_values=[20, 15, 10, 5, 1],
    legend_labels=['20 Injuries', '15 Injuries', '10 Injuries', '5 Injuries', '1 Injury'],
    ax=ax2
)
ax2.set_title("Injurious Crashes in New York City, 2016")

plt.savefig("nyc-collisions-map.png", bbox_inches='tight', pad_inches=0)
