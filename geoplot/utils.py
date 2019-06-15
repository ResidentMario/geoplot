"""
Utilities, principally example data generation algorithms, for use in geoplot testing and
documentation.
"""
import numpy as np
import shapely
import geopandas as gpd
import pandas as pd
try:
    from sklearn.cluster import KMeans
except ImportError:  # Optional dependency, only used for development.
    pass


def gaussian_points(loc=(0, 0), scale=(10, 10), n=100):
    """
    Generates and returns `n` normally distributed points centered at `loc` with `scale` x and y
    directionality.
    """

    arr = np.random.normal(loc, scale, (n, 2))
    return gpd.GeoSeries([shapely.geometry.Point(x, y) for (x, y) in arr])


def classify_clusters(points, n=10):
    """
    Return an array of K-Means cluster classes for an array of `shapely.geometry.Point` objects.
    """
    arr = [[p.x, p.y] for p in points.values]
    clf = KMeans(n_clusters=n)
    clf.fit(arr)
    classes = clf.predict(arr)
    return classes


def gaussian_polygons(points, n=10):
    """
    Returns an array of approximately `n` `shapely.geometry.Polygon` objects for an array of
    `shapely.geometry.Point` objects.
    """
    gdf = gpd.GeoDataFrame(
        data={'cluster_number': classify_clusters(points, n=n)}, geometry=points
    )
    polygons = []
    for i in range(n):
        sel_points = gdf[gdf['cluster_number'] == i].geometry
        polygons.append(shapely.geometry.MultiPoint([(p.x, p.y) for p in sel_points]).convex_hull)
    polygons = [
        p for p in polygons if (
            (not isinstance(p, shapely.geometry.Point)) and
            (not isinstance(p, shapely.geometry.LineString)))
    ]
    return gpd.GeoSeries(polygons)


def gaussian_multi_polygons(points, n=10):
    """
    Returns an array of approximately `n` `shapely.geometry.MultiPolygon` objects for an array of
    `shapely.geometry.Point` objects.
    """
    polygons = gaussian_polygons(points, n*2)
    # Randomly stitch them together.
    polygon_pairs = [
        shapely.geometry.MultiPolygon(list(pair)) for pair in np.array_split(polygons.values, n)
    ]
    return gpd.GeoSeries(polygon_pairs)


def uniform_random_global_points(n=100):
    """
    Returns an array of `n` uniformally distributed `shapely.geometry.Point` objects. Points are
    coordinates distributed equivalently across the Earth's surface.
    """
    xs = np.random.uniform(-180, 180, n)
    ys = np.random.uniform(-90, 90, n)
    return [shapely.geometry.Point(x, y) for x, y in zip(xs, ys)]


def uniform_random_global_network(loc=2000, scale=250, n=100):
    """
    Returns an array of `n` uniformally randomly distributed `shapely.geometry.Point` objects.
    """
    arr = (np.random.normal(loc, scale, n)).astype(int)
    return pd.DataFrame(
        data={
            'mock_variable': arr,
            'from': uniform_random_global_points(n),
            'to': uniform_random_global_points(n)
        }
    )
