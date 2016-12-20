import sys; sys.path.insert(0, '../../')
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as ccrs
import pandas as pd
import matplotlib.pyplot as plt
import shapely

boroughs = gpd.read_file("../../data/nyc_boroughs/boroughs.geojson", driver='GeoJSON')

collisions = pd.read_csv("../../data/nyc_collisions/NYPD_Motor_Vehicle_Collisions.csv", index_col=0)
fatal_collisions = collisions[collisions['BOROUGH'].notnull()]
fatal_collisions = fatal_collisions[fatal_collisions["NUMBER OF PERSONS KILLED"] > 0]

def pointify(srs):
    lat, long = srs['LATITUDE'], srs['LONGITUDE']
    if pd.isnull(lat) or pd.isnull(long):
        return shapely.geometry.Point(0, 0)
    else:
        return shapely.geometry.Point(long, lat)

fatal_collisions = gpd.GeoDataFrame(fatal_collisions,
                                    geometry=fatal_collisions.apply(pointify, axis='columns'))
fatal_collisions = fatal_collisions[fatal_collisions.geometry.map(lambda srs: not (srs.x == 0))]

ax = gplt.polyplot(boroughs, projection=ccrs.AlbersEqualArea())
gplt.pointplot(fatal_collisions, projection=ccrs.AlbersEqualArea(),
               hue='BOROUGH', categorical=True,
               legend=True, edgecolor='white', linewidth=0.5, legend_kwargs={'loc': 'upper left'},
               ax=ax)

plt.savefig("fatal-motor-vehicle-collisions.png")