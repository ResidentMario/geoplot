import shapely.geometry
import geopandas as gpd
import pandas as pd
import numpy as np
from collections import Iterable


class QuadTree:
    def __init__(self, gdf, bounds=None):
        if bounds:
            self.bounds = minx, maxx, miny, maxy = bounds
            gdf = gdf[gdf.geometry.centroid.map(lambda b: (minx < b.x < maxx) & (miny < b.y < maxy))]
        else:
            b = gdf.geometry.bounds
            minx, miny = b[['minx', 'miny']].min().values
            maxx, maxy = b[['maxx', 'maxy']].max().values
            self.bounds = (minx, maxx, miny, maxy)
        gdf = gdf[[not c.is_empty for c in gdf.geometry.centroid]]
        if len(gdf) > 0:
            centroids = gdf.geometry.centroid
            xs, ys = [c.x for c in centroids], [c.y for c in centroids]
            geo = gdf.assign(x=xs, y=ys, centroid=centroids).reset_index()
            groups = geo.groupby(['x', 'y'])
            self.agg = dict()
            self.n = 0
            for ind, subgroup in groups:
                self.n += len(subgroup.index.values)
                self.agg[ind] = subgroup.index.values
            self.data = gdf
        else:
            self.agg = dict()
            self.n = 0
            self.data = gdf

    def split(self):
        # TODO: Investigate why a small number of entries are lost every time this method is run.
        min_x, max_x, min_y, max_y = self.bounds
        mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        q1 = (min_x, mid_x, mid_y, max_y)
        q2 = (min_x, mid_x, min_y, mid_y)
        q3 = (mid_x, max_x, mid_y, max_y)
        q4 = (mid_x, max_x, min_y, mid_y)
        return [QuadTree(self.data, bounds=q1), QuadTree(self.data, bounds=q2), QuadTree(self.data, bounds=q3),
                QuadTree(self.data, bounds=q4)]


def partition(quadtree, thresh):
    if quadtree.n < thresh:
        return [quadtree]
    else:
        ret = subpartition(quadtree, thresh)
        return flatten(ret)


def subpartition(quadtree, thresh):
    subtrees = quadtree.split()
    if any([t.n < thresh for t in subtrees]):
        return [quadtree]
    else:
        return [partition(q, thresh) for q in subtrees]


def flatten(items):
    """
    Yield items from any nested iterable; see REF.
    cf. http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    """
    for x in items:
        if isinstance(x, Iterable):
            yield from flatten(x)
        else:
            yield x
