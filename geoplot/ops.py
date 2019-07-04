"""
This module implements a naive equal-split four-way quadtree algorithm 
(https://en.wikipedia.org/wiki/Quadtree). It has been written in way meant to make it convenient
to use for splitting and aggregating rectangular geometries up to a certain guaranteed minimum
instance threshold.

The routines here are used by the ``geoplot.quadtree`` plot type.
"""

from collections import Iterable
import geopandas as gpd
import numpy as np
import pandas as pd
import shapely


class QuadTree:
    """
    This class implements a naive equal-split four-way quadtree algorithm
    (https://en.wikipedia.org/wiki/Quadtree). It has been written in way meant to make it
    convenient to use for splitting and aggregating rectangular geometries up to a certain
    guaranteed minimum instance threshold.

    Properties
    ----------
    data : GeoDataFrame
        An efficient shallow copy reference to the class's ``gdf`` data initialization input.
        This is retained for downstream aggregation purposes.
    bounds : (minx, maxx, miny, maxy)
        A tuple of boundaries for data contained in the quadtree. May be passed as an
        initialization input via ``bounds`` or left to the ``QuadTree`` instance to compute for
        itself.
    agg : dict
        An aggregated dictionary whose keys consist of coordinates within the instance's 
        ``bounds`` and whose values consist of the indices of rows in the ``data`` property
        corresponding with those points. This additional bookkeeping is necessary because a 
        single coordinate may contain many individual data points.
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
            gdf = gdf[
                gdf.geometry.map(lambda b: (minx < b.x < maxx) & (miny < b.y < maxy))
            ]
        else:
            b = gdf.geometry.bounds
            minx, miny = b[['minx', 'miny']].min().values
            maxx, maxy = b[['maxx', 'maxy']].max().values
            self.bounds = (minx, maxx, miny, maxy)
        gdf = gdf[[not p.is_empty for p in gdf.geometry]]
        if len(gdf) > 0:
            points = gdf.geometry
            xs, ys = [p.x for p in points], [p.y for p in points]
            geo = gpd.GeoDataFrame(index=gdf.index).assign(x=xs, y=ys).reset_index()
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
        A list of four "sub" QuadTree instances, corresponding with the first, second, third, and
        fourth quartiles, respectively.
        """
        # TODO: Investigate why a small number of entries are lost every time this method is run.
        min_x, max_x, min_y, max_y = self.bounds
        mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        q1 = (min_x, mid_x, mid_y, max_y)
        q2 = (min_x, mid_x, min_y, mid_y)
        q3 = (mid_x, max_x, mid_y, max_y)
        q4 = (mid_x, max_x, min_y, mid_y)
        return [
            QuadTree(self.data, bounds=q1),
            QuadTree(self.data, bounds=q2), 
            QuadTree(self.data, bounds=q3),
            QuadTree(self.data, bounds=q4)
        ]

    def partition(self, nmin, nmax):
        """
        This method call decomposes a QuadTree instances into a list of sub- QuadTree instances
        which are the smallest possible geospatial "buckets", given the current splitting rules,
        containing at least ``thresh`` points.

        Parameters
        ----------
        thresh : int
            The minimum number of points per partition. Care should be taken not to set this
            parameter to be too low, as in large datasets a small cluster of highly adjacent
            points may result in a number of sub-recursive splits possibly in excess of Python's
            global recursion limit.

        Returns
        -------
        partitions : list of QuadTree object instances
            A list of sub- QuadTree instances which are the smallest possible geospatial
            "buckets", given the current splitting rules, containing at least ``thresh`` points.
        """
        if self.n < nmin:
            return [self]
        else:
            ret = self.subpartition(nmin, nmax)
            return self.flatten(ret)

    def subpartition(self, nmin, nmax):
        """
        Recursive core of the ``QuadTree.partition`` method. Just five lines of code, amazingly.

        Parameters
        ----------
        quadtree : QuadTree object instance
            The QuadTree object instance being partitioned.
        nmin : int
            The splitting threshold. If this is not met this method will return a listing
            containing the root tree alone.

        Returns
        -------
        A (probably nested) list of QuadTree object instances containing a number of points
        respecting the threshold parameter.
        """
        subtrees = self.split()
        if self.n > nmax:
            return [q.partition(nmin, nmax) for q in subtrees]
        elif any([t.n < nmin for t in subtrees]):
            return [self]
        else:
            return [q.partition(nmin, nmax) for q in subtrees]

    @staticmethod
    def flatten(items):
        """
        Yield items from any nested iterable. Used by ``QuadTree.flatten`` to one-dimensionalize a
        list of sublists. cf.
        http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
        """
        for x in items:
            if isinstance(x, Iterable):
                yield from QuadTree.flatten(x)
            else:
                yield x


def build_voronoi_polygons(df):
    """
    Given a GeoDataFrame of point geometries and pre-computed plot extrema, build Voronoi
    simplexes for the given points in the given space and returns them.

    Voronoi simplexes which are located on the edges of the graph may extend into infinity in some
    direction. In other words, the set of points nearest the given point does not necessarily have
    to be a closed polygon. We force these non-hermetic spaces into polygons using a subroutine.

    Returns a list of shapely.geometry.Polygon objects, each one a Voronoi polygon.
    """
    from scipy.spatial import Voronoi
    geom = np.array(df.geometry.map(lambda p: [p.x, p.y]).tolist())
    vor = Voronoi(geom)

    polygons = []

    for idx_point, _ in enumerate(vor.points):
        idx_point_region = vor.point_region[idx_point]
        idxs_vertices = np.array(vor.regions[idx_point_region])

        is_finite = not np.any(idxs_vertices == -1)

        if is_finite:
            # Easy case, the region is closed. Make a polygon out of the Voronoi ridge points.
            idx_point_region = vor.point_region[idx_point]
            idxs_vertices = np.array(vor.regions[idx_point_region])
            region_vertices = vor.vertices[idxs_vertices]
            region_poly = shapely.geometry.Polygon(region_vertices)
            polygons.append(region_poly)

        else:
            # Hard case, the region is open. Project new edges out to the margins of the plot.
            # See `scipy.spatial.voronoi_plot_2d` for the source of this calculation.
            point_idx_ridges_idx = np.where((vor.ridge_points == idx_point).any(axis=1))[0]

            # TODO: why does this happen?
            if len(point_idx_ridges_idx) == 0:
                continue

            ptp_bound = vor.points.ptp(axis=0)
            center = vor.points.mean(axis=0)

            finite_segments = []
            infinite_segments = []

            pointwise_ridge_points = vor.ridge_points[point_idx_ridges_idx]
            pointwise_ridge_vertices = np.asarray(vor.ridge_vertices)[point_idx_ridges_idx]

            for pointidx, simplex in zip(pointwise_ridge_points, pointwise_ridge_vertices):
                simplex = np.asarray(simplex)

                if np.all(simplex >= 0):
                    finite_segments.append(vor.vertices[simplex])

                else:
                    i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                    t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[pointidx].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = vor.vertices[i] + direction * ptp_bound.max()

                    infinite_segments.append(np.asarray([vor.vertices[i], far_point]))

            finite_segments = finite_segments if finite_segments else np.zeros(shape=(0,2,2))
            ls = np.vstack([np.asarray(infinite_segments), np.asarray(finite_segments)])

            # We have to trivially sort the line segments into polygonal order. The algorithm that
            # follows is inefficient, being O(n^2), but "good enough" for this use-case.
            ls_sorted = []

            while len(ls_sorted) < len(ls):
                l1 = ls[0] if len(ls_sorted) == 0 else ls_sorted[-1]
                matches = []

                for l2 in [l for l in ls if not (l == l1).all()]:
                    if np.any(l1 == l2):
                        matches.append(l2)
                    elif np.any(l1 == l2[::-1]):
                        l2 = l2[::-1]
                        matches.append(l2)

                if len(ls_sorted) == 0:
                    ls_sorted.append(l1)

                for match in matches:
                    # in list sytax this would be "if match not in ls_sorted"
                    # in numpy things are more complicated...
                    if not any((match == ls_sort).all() for ls_sort in ls_sorted):
                        ls_sorted.append(match)
                        break

            # Build and return the final polygon.
            polyline = np.vstack(ls_sorted)
            geom = shapely.geometry.Polygon(polyline).convex_hull
            polygons.append(geom)

    return polygons


def jitter_points(geoms):
    working_df = gpd.GeoDataFrame().assign(
        _x=geoms.x,
        _y=geoms.y,
        geometry=geoms
    )
    group = working_df.groupby(['_x', '_y'])
    group_sizes = group.size()

    if not (group_sizes > 1).any():
        return geoms

    else:
        jitter_indices = []

        group_indices = group.indices
        group_keys_of_interest = group_sizes[group_sizes > 1].index
        for group_key_of_interest in group_keys_of_interest:
            jitter_indices += group_indices[group_key_of_interest].tolist()

        _x_jitter = (
            pd.Series([0] * len(working_df)) +
            pd.Series(
                ((np.random.random(len(jitter_indices)) - 0.5)  * 10**(-5)),
                index=jitter_indices
            )
        )
        _x_jitter = _x_jitter.fillna(0)

        _y_jitter = (
            pd.Series([0] * len(working_df)) +
            pd.Series(
                ((np.random.random(len(jitter_indices)) - 0.5)  * 10**(-5)),
                index=jitter_indices
            )
        )
        _y_jitter = _y_jitter.fillna(0)

        out = gpd.GeoSeries([
            shapely.geometry.Point(x, y) for x, y in
            zip(working_df._x + _x_jitter, working_df._y + _y_jitter)
        ])

        # guarantee that no two points have the exact same coordinates
        regroup_sizes = (
            gpd.GeoDataFrame()
            .assign(_x=out.x, _y=out.y)
            .groupby(['_x', '_y'])
            .size()
        )
        assert not (regroup_sizes > 1).any()

        return out
