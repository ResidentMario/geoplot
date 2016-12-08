"""
Utilities, principally example data generation algorithms, for use in geoplot testing and documentation.
"""
import numpy as np
import shapely
import geopandas as gpd
from sklearn.cluster import KMeans


def gaussian_points(loc=(0, 0), scale=(10, 10), n=100):
    arr = np.random.normal(loc, scale, (n, 2))
    return gpd.GeoSeries([shapely.geometry.Point(x, y) for (x, y) in arr])


def classify_clusters(points, n=10):
    arr = [[p.x, p.y] for p in points.values]
    clf = KMeans(n_clusters=n)
    clf.fit(arr)
    classes = clf.predict(arr)
    return classes


def gaussian_polygons(points, n=10):
    gdf = gpd.GeoDataFrame(data={'cluster_number': classify_clusters(points, n=n)}, geometry=points)
    polygons = []
    for i in range(n):
        sel_points = gdf[gdf['cluster_number'] == i].geometry
        polygons.append(shapely.geometry.MultiPoint([(p.x, p.y) for p in sel_points]).convex_hull)
    return gpd.GeoSeries(polygons)