"""
This module implements a naive equal-split four-way quadtree algorithm (https://en.wikipedia.org/wiki/Quadtree). It
has been written in way meant to make it convenient to use for splitting and aggregating rectangular geometries up
to a certain guaranteed minimum instance threshold.

The routines here are used by the ``geoplot.aggplot`` plot type, and only when no user geometry input is provided.
"""

from collections import Iterable


class QuadTree:
    """
    This module's core class. For more on quadtrees cf. https://en.wikipedia.org/wiki/Quadtree.

    Properties
    ----------
    data : GeoDataFrame
        An efficient shallow copy reference to the class's ``gdf`` data initialization input. This is retained for
        downstream aggregation purposes.
    bounds : (minx, maxx, miny, maxy)
        A tuple of boundaries for data contained in the quadtree. May be passed as an initialization input via
        ``bounds`` or left to the ``QuadTree`` instance to compute for itself.
    agg : dict
        An aggregated dictionary whose keys consist of coordinates within the instance's ``bounds`` and whose values
        consist of the indices of rows in the ``data`` property corresponding with those points. This additional
        bookkeeping is necessary because a single coordinate may contain many individual data points.
    n : int
        The number of points contained in the current QuadTree instance.
    """
    def __init__(self, gdf, bounds=None):
        """
        Instantiation method.

        Parameters
        ----------
        gdf : GeoDataFrame
            The data being geospatially aggregated.
        bounds : None or (minx, maxx, miny, maxy), optional
            Precomputed extrema of the ``gdf`` input. If not provided beforehand

        Returns
        -------
        A baked `QuadTree` class instance.
        """
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
        """
        Splits the current QuadTree instance four ways through the midpoint.

        Returns
        -------
        A list of four "sub" QuadTree instances, corresponding with the first, second, third, and fourth quartiles,
        respectively.
        """
        # TODO: Investigate why a small number of entries are lost every time this method is run.
        min_x, max_x, min_y, max_y = self.bounds
        mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        q1 = (min_x, mid_x, mid_y, max_y)
        q2 = (min_x, mid_x, min_y, mid_y)
        q3 = (mid_x, max_x, mid_y, max_y)
        q4 = (mid_x, max_x, min_y, mid_y)
        return [QuadTree(self.data, bounds=q1), QuadTree(self.data, bounds=q2), QuadTree(self.data, bounds=q3),
                QuadTree(self.data, bounds=q4)]

    def partition(self, nmin, nmax):
        """
        This method call decomposes a QuadTree instances into a list of sub- QuadTree instances which are the
        smallest possible geospatial "buckets", given the current splitting rules, containing at least ``thresh``
        points.

        Parameters
        ----------
        thresh : int
            The minimum number of points per partition. Care should be taken not to set this parameter to be too
            low, as in large datasets a small cluster of highly adjacent points may result in a number of
            sub-recursive splits possibly in excess of Python's global recursion limit.

        Returns
        -------
        partitions : list of QuadTree object instances
            A list of sub- QuadTree instances which are the smallest possible geospatial "buckets", given the current
            splitting rules, containing at least ``thresh`` points.
        """
        if self.n < nmin:
            return [self]
        else:
            ret = subpartition(self, nmin, nmax)
            return flatten(ret)


def subpartition(quadtree, nmin, nmax):
    """
    Recursive core of the ``QuadTree.partition`` method. Just five lines of code, amazingly.

    Parameters
    ----------
    quadtree : QuadTree object instance
        The QuadTree object instance being partitioned.
    nmin : int
        The splitting threshold. If this is not met this method will return a listing containing the root tree alone.

    Returns
    -------
    A (probably nested) list of QuadTree object instances containing a number of points respecting the threshold
    parameter.
    """
    subtrees = quadtree.split()
    if quadtree.n > nmax:
        return [q.partition(nmin, nmax) for q in subtrees]
    elif any([t.n < nmin for t in subtrees]):
        return [quadtree]
    else:
        return [q.partition(nmin, nmax) for q in subtrees]


def flatten(items):
    """
    Yield items from any nested iterable. Used by ``QuadTree.flatten`` to one-dimensionalize a list of sublists.
    cf. http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
    """
    for x in items:
        if isinstance(x, Iterable):
            yield from flatten(x)
        else:
            yield x
