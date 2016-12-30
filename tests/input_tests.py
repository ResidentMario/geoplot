"""
This test file runs static tests of argument inputs for arguments of the various plot types which accept a variety
of input formats. When the library passes this test, that ought to mean that all of the input formats supported and
listed in the API Reference work.
"""

import sys; sys.path.insert(0, '../')
import geoplot as gplt
import unittest
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import numpy as np
import pandas as pd


# Point-type DataFrame input.
list_gaussian_points = gplt.utils.gaussian_points(n=4)
series_gaussian_points = gpd.GeoSeries(list_gaussian_points)
dataframe_gaussian_points = gpd.GeoDataFrame(geometry=series_gaussian_points).assign(hue_var=[1,2,3,4])


# Polygon-type DataFrame input.
list_gaussian_polys = gplt.utils.gaussian_polygons(gplt.utils.gaussian_points(n=1000), n=2).append(
                 gplt.utils.gaussian_multi_polygons(gplt.utils.gaussian_points(n=1000), n=2)
)
series_gaussian_polys = gpd.GeoSeries(list_gaussian_polys)
dataframe_gaussian_polys = gpd.GeoDataFrame(geometry=series_gaussian_polys).assign(hue_var=[1,2,3,4])

# Hue input.
list_hue_values = [1, 2, 3, 4]
series_hue_values = pd.Series(list_hue_values)
def map_hue_values(): return map(lambda i: list_hue_values[i], list(range(len(list_hue_values))))

# (Sankey) point input.
# Start and End variables.
list_start_points = [Point(a + 2, a - 2) for a in range(0, 4)]
list_end_points = [Point(a - 2, a + 2) for a in range(1, 5)]
series_start_points, series_end_points = gpd.GeoSeries(list_start_points), gpd.GeoSeries(list_end_points)
def map_start_points(): return map(lambda i: list_start_points[i], list(range(len(list_start_points))))
def map_end_points(): return map(lambda i: list_end_points[i], list(range(len(list_end_points))))
dataframe_gaussian_points = dataframe_gaussian_points.assign(starts=list_start_points, ends=list_end_points)

# (Sankey) paths.
list_paths = [LineString([[a.x, a.y], [b.x, b.y]]) for a, b in zip(list_start_points, list_end_points)]
series_paths = gpd.GeoSeries(list_paths)
def map_paths(): return map(lambda i: list_paths[i], list(range(len(list_paths))))
dataframe_gaussian_points = dataframe_gaussian_points.assign(paths=list_paths)

# (Aggplot) geometry.
dataframe_gaussian_points = dataframe_gaussian_points.assign(mock_category=np.random.randint(1, 5))
aggplot_geometries = dataframe_gaussian_polys.set_index('hue_var', drop=True)


class TestDataInputFormats(unittest.TestCase):

    def test_pointplot(self):
        try:
            gplt.pointplot(series_gaussian_points, k=2)
            gplt.pointplot(dataframe_gaussian_points, k=2)

            gplt.pointplot(dataframe_gaussian_points, hue=list_hue_values, k=None)
            gplt.pointplot(dataframe_gaussian_points, hue=series_hue_values, k=None)
            gplt.pointplot(dataframe_gaussian_points, hue=map_hue_values(), k=None)
            gplt.pointplot(dataframe_gaussian_points, hue='hue_var', k=None)
        finally: plt.close()

    def test_kdeplot(self):
        try:
            gplt.kdeplot(series_gaussian_points)
            gplt.kdeplot(dataframe_gaussian_points)

            gplt.kdeplot(dataframe_gaussian_points, hue=list_hue_values)
            gplt.kdeplot(dataframe_gaussian_points, hue=series_hue_values)
            gplt.kdeplot(dataframe_gaussian_points, hue=map_hue_values)
            gplt.kdeplot(dataframe_gaussian_points, hue='hue_var')
        finally:
            plt.close()

    def test_cartogram(self):
        try:
            gplt.cartogram(series_gaussian_polys, scale=list_hue_values)
            gplt.cartogram(dataframe_gaussian_polys, scale=list_hue_values)

            gplt.cartogram(dataframe_gaussian_polys, hue=list_hue_values, scale=list_hue_values)
            gplt.cartogram(dataframe_gaussian_polys, hue=series_hue_values, scale=list_hue_values)
            gplt.cartogram(dataframe_gaussian_polys, hue=map_hue_values(), scale=list_hue_values)
            gplt.cartogram(dataframe_gaussian_polys, hue='hue_var', scale=list_hue_values)
        finally:
            plt.close()

    def test_polyplot(self):
        try:
            gplt.polyplot(series_gaussian_polys)
            gplt.polyplot(dataframe_gaussian_polys)

        finally:
            plt.close()

    def test_choropleth(self):
        try:
            gplt.choropleth(series_gaussian_polys, hue=list_hue_values)
            gplt.choropleth(dataframe_gaussian_polys, hue=list_hue_values)

            gplt.choropleth(dataframe_gaussian_polys, hue=list_hue_values)
            gplt.choropleth(dataframe_gaussian_polys, hue=series_hue_values)
            gplt.choropleth(dataframe_gaussian_polys, hue=map_hue_values())
            gplt.choropleth(dataframe_gaussian_polys, hue='hue_var')
        finally:
            plt.close()

    def test_sankey(self):
        try:
            gplt.sankey(start=map_start_points(), end=map_end_points())
            gplt.sankey(start=map_start_points(), end=map_end_points())

            gplt.sankey(start=list_start_points, end=list_end_points)
            gplt.sankey(start=list_start_points, end=list_end_points)

            gplt.sankey(start=series_start_points, end=series_end_points)
            gplt.sankey(start=series_start_points, end=series_end_points)

            gplt.sankey(start=map_start_points(), end=map_end_points())
            gplt.sankey(start=map_start_points(), end=map_end_points())

            gplt.sankey(dataframe_gaussian_points, start='starts', end='ends')

            gplt.sankey(path=list_paths)
            gplt.sankey(path=series_paths)
            gplt.sankey(path=map_paths())

            gplt.sankey(dataframe_gaussian_points, path='paths')

        finally:
            plt.close()

    def test_aggplot(self):
        try:
            gplt.aggplot(series_gaussian_points, hue=list_hue_values)
            gplt.aggplot(dataframe_gaussian_points, hue=list_hue_values)

            gplt.aggplot(dataframe_gaussian_points, hue=list_hue_values)
            gplt.aggplot(dataframe_gaussian_points, hue=series_hue_values)
            gplt.aggplot(dataframe_gaussian_points, hue=map_hue_values())
            gplt.aggplot(dataframe_gaussian_points, hue='hue_var')

            gplt.aggplot(dataframe_gaussian_points, hue=list_hue_values, by='mock_category')
            gplt.aggplot(dataframe_gaussian_points, hue=list_hue_values,
                         by=dataframe_gaussian_points['mock_category'])  # Series
            gplt.aggplot(dataframe_gaussian_points, hue=list_hue_values,
                         by=list(dataframe_gaussian_points['mock_category']))  # List
            gplt.aggplot(dataframe_gaussian_points, hue=list_hue_values,
                         by=map(lambda v: v, list(dataframe_gaussian_points['mock_category'])))  # Map

            gplt.aggplot(dataframe_gaussian_points, hue=list_hue_values, by='mock_category',
                         geometry=aggplot_geometries)
            gplt.aggplot(dataframe_gaussian_points, hue=list_hue_values,
                         by=dataframe_gaussian_points['mock_category'],
                         geometry=aggplot_geometries)  # Series
            gplt.aggplot(dataframe_gaussian_points, hue=list_hue_values,
                         by=list(dataframe_gaussian_points['mock_category']),
                         geometry=aggplot_geometries)  # List
            gplt.aggplot(dataframe_gaussian_points, hue=list_hue_values,
                         by=map(lambda v: v, list(dataframe_gaussian_points['mock_category'])),
                         geometry=aggplot_geometries)  # Map

        finally:
            plt.close()
