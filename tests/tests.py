"""
Tests.
"""

import geoplot as gplt
import geoplot.crs as gcrs
import unittest
import hypothesis
from hypothesis import given
import hypothesis.strategies as hyp
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import matplotlib as mpl
import cartopy
import cartopy.crs as ccrs
import shapely


class TestCoreMethods(unittest.TestCase):

    # def test_init_figure(self):
    #     gplt._init_figure(None, (8, 6))  # Case 1: No axis is passed.
    #     gplt._init_figure(plt.gca(), (8, 6))  # Case 2: An AxesSubplot is passed.
    #     gplt._init_figure(plt.axes(projection=ccrs.PlateCarree()), (8, 6))  # Case 3: A GeoAxesSubplot is passed.
    #
    # @given(s=hyp.lists(hyp.tuples(hyp.floats(allow_nan=False, allow_infinity=False),
    #                               hyp.floats(allow_nan=False, allow_infinity=False))).map(
    #     lambda coords: shapely.geometry.Polygon(coords).envelope
    # ).example())
    # def test_get_envelopes_min_maxes(self, s):
    #     gplt._get_envelopes_min_maxes(s)

    pass
