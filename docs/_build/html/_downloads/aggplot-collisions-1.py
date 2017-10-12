# Load the data.
from quilt.data.ResidentMario import nyc_collisions
import shapely
import geopandas as gpd
import pandas as pd

collisions = nyc_collisions.collisions()

def pointify(srs):
    try:
        lat, long = float(srs['LATITUDE']), float(srs['LONGITUDE'])
        if pd.isnull(lat) or pd.isnull(long):
            return shapely.geometry.Point((0, 0))
        else:
            return shapely.geometry.Point((long, lat))
    except ValueError:
        return shapely.geometry.Point((0, 0))

collisions = gpd.GeoDataFrame(collisions.head(5000), geometry=collisions.head(5000).apply(pointify, axis='columns'))
collisions = collisions[collisions.geometry.map(lambda srs: not (srs.x == 0))]
collisions['BOROUGH'] = collisions['BOROUGH'].str.strip().map(lambda v: np.nan if len(v) == 0 else v)


# Plot the data.
import geoplot as gplt
import geoplot.crs as gcrs
import numpy as np
import matplotlib.pyplot as plt

f, axarr = plt.subplots(3, 1, figsize=(12, 12), subplot_kw={
    'projection': gcrs.AlbersEqualArea(central_latitude=40.7128, central_longitude=-74.0059)
})
plt.suptitle('Max(Injuries) in Collision by Area, 2016', fontsize=16)
plt.subplots_adjust(top=0.95)


# In the first plot we do not provide any geographic data at all as input. In this case aggplot takes the centroids
# of whatever it is that we are throwing at it and uses them to decompose the boundaries of our data into squares,
# with a cetain user specified minimum (nmin) and maximum (nmax) number of observations per square. This is known in
# the literature as a QuadTree. An additional parameter, nsig, controls how many observations have to be made in a
# square for that square to be considered significant (insignificant and empty squares are not colored in). The agg
# parameter controls the method by which the observations are aggregated---in the default case np.mean is used,
# in this case we have specified a maximum (np.max) instead.
ax1 = gplt.aggplot(collisions, projection=gcrs.AlbersEqualArea(),
                   hue='NUMBER OF PERSONS INJURED', agg=np.max,
                   nmin=100, nmax=500, cmap='Reds', linewidth=0.5, edgecolor='white', ax=axarr[0])
ax1.set_title("No Geometry (Quadtree)")


# In the second plot we provide more information than the first, by specifying a categorical column of data in the
# dataset corresponding with sort of encoded geography---in this example, the postal zip code. Aggplot computes the
# geometries it needs itself, using a simple convex hull around the observations' point cloud. Albeit not elegant,
# the resulting geometry is functional---and, again, spares us the task of having to find our own.
ax2 = gplt.aggplot(collisions, projection=gcrs.AlbersEqualArea(),
                   hue='NUMBER OF PERSONS INJURED', agg=np.max, by='ZIP CODE',
                   cmap='Reds', linewidth=0.5, edgecolor='white', ax=axarr[1])
ax2.set_title("Categorical Geometry (Convex Hull)")


# In the third plot we finally provide our own geometry---again, in this example, a GeoSeries of zip codes we found
# somewhere else on the Internet. In this case the result snaps into focus quite clearly, and is equivalent in form
# to the geoplot.choropleth facility (with k=None; the later provides more options but doesn't aggregate geometries
# for you, however).
zip_codes = gplt.datasets.load('nyc-zip-codes')
ax3 = gplt.aggplot(collisions, projection=gcrs.AlbersEqualArea(),
                   hue='NUMBER OF PERSONS INJURED', agg=np.max,
                   by='ZIP CODE', geometry=zip_codes.geometry,
                   cmap='Reds', linewidth=0.5, edgecolor='white', ax=axarr[2])
ax3.set_title("Geometry Provided (Choropleth)")


plt.savefig("aggplot-collisions-1.png", bbox_inches='tight', pad_inches=0.1)