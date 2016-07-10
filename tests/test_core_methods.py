"""
Tests for core methods (meaning basically methods with _ or __ before their name).
"""

import unittest
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
from pandas import DataFrame, Series
import geoplot.core


class TestCoreMethods(unittest.TestCase):

    def setUp(self):
        pass

    # def _test__cast_as_geodataframe(self):
    #     """
    #     Tests the __cast_as_geodataframe core method.
    #     """
    #     # GeoDataFrame test.
    #     # In this case the method naively returns the input.
    #     gdf = gpd.read_file(gpd.datasets.get_path(gpd.datasets.available[0])) # World cities geopandas test dataset.
    #     gdf_c = geoplot.core.__cast_as_geodataframe(gdf)
    #     self.assertIsInstance(gdf_c, GeoDataFrame)
    #
    #     # GeoSeries test.
    #     gs = gdf['geometry']
    #     gdf_c = geoplot.core.__cast_as_geodataframe(gs)
    #     self.assertIsInstance(gdf_c, GeoDataFrame)
    #
    #     # Series test.
    #     # s = gdf['geometry']
    #     # self.assertEqual(gdf._geometry_column_name, geometry_column_name)