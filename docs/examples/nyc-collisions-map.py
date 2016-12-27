import sys; sys.path.insert(0, '../../')
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import pandas as pd
import matplotlib.pyplot as plt
from shapely.geometry import Point

# Shape the data.
boroughs = gpd.read_file("../../data/nyc_boroughs/boroughs.geojson", driver='GeoJSON')

collisions = pd.read_csv("../../data/nyc_collisions/NYPD_Motor_Vehicle_Collisions_2016.csv")
collisions = collisions[collisions['BOROUGH'].notnull()]

fatal_collisions = collisions[collisions["NUMBER OF PERSONS KILLED"] > 0]
injurious_collisions = collisions[collisions["NUMBER OF PERSONS INJURED"] > 0]


def pointify(srs):
    lat, long = srs['LATITUDE'], srs['LONGITUDE']
    if pd.isnull(lat) or pd.isnull(long):
        return Point(0, 0)
    else:
        return Point(long, lat)

fatal_collisions = gpd.GeoDataFrame(fatal_collisions,
                                    geometry=fatal_collisions.apply(pointify, axis='columns'))
fatal_collisions = fatal_collisions[fatal_collisions.geometry.map(lambda srs: not (srs.x == 0))]
fatal_collisions = fatal_collisions[fatal_collisions['DATE'].map(lambda day: "2016" in day)]

injurious_collisions = gpd.GeoDataFrame(injurious_collisions,
                                        geometry=injurious_collisions.apply(pointify, axis='columns'))
injurious_collisions = injurious_collisions[injurious_collisions.geometry.map(lambda srs: not (srs.x == 0))]
injurious_collisions = injurious_collisions[injurious_collisions['DATE'].map(lambda day: "2016" in day)]


# Plot the data.
fig = plt.figure(figsize=(10,5))

ax1 = plt.subplot(121, projection=gcrs.AlbersEqualArea(central_latitude=40.7128, central_longitude=-74.0059))
gplt.polyplot(boroughs, ax=ax1, projection=gcrs.AlbersEqualArea())
gplt.pointplot(fatal_collisions, projection=gcrs.AlbersEqualArea(),
               hue='BOROUGH', categorical=True,
               edgecolor='white', linewidth=0.5, zorder=10,
               scale='NUMBER OF PERSONS KILLED', limits=(2, 8),
               legend=True, legend_var='scale', legend_kwargs={'loc': 'upper left'},
               legend_values=[2, 1], legend_labels=['2 Fatalities', '1 Fatality'],
               ax=ax1)
plt.title("Fatal Crashes in New York City, 2016")

ax2 = plt.subplot(122, projection=gcrs.AlbersEqualArea(central_latitude=40.7128, central_longitude=-74.0059))
gplt.polyplot(boroughs, ax=ax2, projection=gcrs.AlbersEqualArea())
gplt.pointplot(injurious_collisions, projection=gcrs.AlbersEqualArea(),
               hue='BOROUGH', categorical=True,
               edgecolor='white', linewidth=0.5, zorder=10,
               scale='NUMBER OF PERSONS INJURED', limits=(1, 10),
               legend=True, legend_var='scale', legend_kwargs={'loc': 'upper left'},
               legend_values=[20, 15, 10, 5, 1],
               legend_labels=['20 Injuries', '15 Injuries', '10 Injuries', '5 Injuries', '1 Injury'],
               ax=ax2)
plt.title("Injurious Crashes in New York City, 2016")


plt.savefig("nyc-collisions-map.png", bbox_inches='tight', pad_inches=0)