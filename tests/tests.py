"""
Testing assumptions:

1. Data input is pre-validated.
2. Projections besides the one used for testing projections work equivalently to the one that is being tested.
3. Extent works as expected (not true: https://github.com/ResidentMario/geoplot/issues/21).
"""

import sys; sys.path.insert(0, '../')
import geoplot as gplt
import geoplot.crs as gcrs
import unittest
import hypothesis
from hypothesis import given
import hypothesis.strategies as hyp
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy
import cartopy.crs as ccrs
import shapely
import random


# # Define strategies.
# # Points.
# x_point = hyp.floats(min_value=-180, max_value=180, allow_nan=False, allow_infinity=False)
# y_point = hyp.floats(min_value=-90, max_value=90, allow_nan=False, allow_infinity=False)
# coordinates = hyp.builds(shapely.geometry.Point, x_point, y_point)
# coordinate_lists = hyp.lists(coordinates, min_size=1)
# point_geoseries_many = hyp.builds(gpd.GeoSeries, coordinate_lists)
#
# # (Valid) polygons.
# polygonal_coordinate_lists = hyp.lists(hyp.tuples(x_point, y_point), min_size=3, unique=True)
# polygon = hyp.builds(shapely.geometry.Polygon, polygonal_coordinate_lists)\
#             .map(lambda poly: poly.buffer(0))\
#             .filter(lambda poly: not poly.is_empty)\
#             .map(lambda poly: poly.buffer(10))
# polygon_lists = hyp.lists(polygon, min_size=1)
# polygon_geoseries_many = hyp.builds(gpd.GeoSeries, polygon_lists)

# (Valid) multipolygons.

# @hyp.composite
# def polygon(draw):
#     polygonal_segment = polygonal_coordinate_lists()
#     name = draw(names)
#     date1 = draw(project_date)
#     date2 = draw(project_date)
#     assume(date1 != date2)
#     start = min(date1, date2)
#     end = max(date1, date2)
#     return Project(name, start, en

# Projections
projections = hyp.sampled_from((None, gcrs.AlbersEqualArea()))


class TestPointPlot(unittest.TestCase):

    @given(point_geoseries_many, projections)
    def test_point_plot(self, point_geoseries, projection):
        gplt.pointplot(point_geoseries, projection=projection, extent=(-180, 180, -90, 90))
        plt.close()


class TestPolyPlot(unittest.TestCase):

    @given(polygon_geoseries_many, projections)
    def test_point_plot(self, polygon_geoseries, projection):
        gplt.polyplot(polygon_geoseries, projection=projection, extent=(-180, 180, -90, 90))
        plt.close()