"""
This test file runs static tests keyword argument inputs for types underlying geoplot functions.
"""

import sys; sys.path.insert(0, '../')
import geoplot as gplt
import unittest
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, LineString
import numpy as np
import pandas as pd
import geoplot.crs as gcrs


# Point-type DataFrame input.
list_gaussian_points = gplt.utils.gaussian_points(n=4)
dataframe_gaussian_points = gpd.GeoDataFrame(geometry=list_gaussian_points).assign(hue_var=[1,2,3,4])


# Polygon-type DataFrame input.
list_gaussian_polys = gplt.utils.gaussian_polygons(gplt.utils.gaussian_points(n=1000), n=2).append(
                 gplt.utils.gaussian_multi_polygons(gplt.utils.gaussian_points(n=1000), n=2)
)
dataframe_gaussian_polys = gpd.GeoDataFrame(geometry=list_gaussian_polys).assign(hue_var=[1,2,3,4])


# # Hue input.
# list_hue_values = [1, 2, 3, 4]

# (Sankey) point input.
# Start and End variables.
list_start_points = [Point(a + 2, a - 2) for a in range(0, 4)]
list_end_points = [Point(a - 2, a + 2) for a in range(1, 5)]

# (Sankey) paths.
list_paths = [LineString([[a.x, a.y], [b.x, b.y]]) for a, b in zip(list_start_points, list_end_points)]

# (Aggplot) geometry.
dataframe_gaussian_points = dataframe_gaussian_points.assign(mock_category=np.random.randint(1, 5))
aggplot_geometries = dataframe_gaussian_polys.set_index('hue_var', drop=True)


class TestDataInputFormats(unittest.TestCase):

    def test_pointplot(self):
        try:
            gplt.pointplot(list_gaussian_points, color='white')
            gplt.pointplot(list_gaussian_points, projection=gcrs.PlateCarree(), color='white')

            gplt.pointplot(list_gaussian_points, s=5)
            gplt.pointplot(list_gaussian_points, projection=gcrs.PlateCarree(), s=5)

            gplt.pointplot(list_gaussian_points, legend_kwargs={'fancybox': False})
            gplt.pointplot(list_gaussian_points, projection=gcrs.PlateCarree(), legend_kwargs={'fancybox': False})
        finally: plt.close()

    def test_kdeplot(self):
        # All keyword arguments are passed directly to KDEPlot and not mutated.
        pass

    def test_polyplot(self):
        try: gplt.polyplot(list_gaussian_polys, color='white')
        finally: plt.close()

    def test_choropleth(self):
        try:
            gplt.choropleth(dataframe_gaussian_polys, hue='hue_var', legend_kwargs={'fancybox': False})
            gplt.choropleth(dataframe_gaussian_polys, hue='hue_var',
                            projection=gcrs.PlateCarree(), legend_kwargs={'fancybox': False})
        finally: plt.close()

    def test_aggplot(self):
        try:
            gplt.aggplot(dataframe_gaussian_points, hue='mock_category')
            gplt.aggplot(dataframe_gaussian_points, hue='mock_category', projection=gcrs.PlateCarree())

            gplt.aggplot(dataframe_gaussian_points, hue='mock_category', by='mock_category')
            gplt.aggplot(dataframe_gaussian_points, hue='mock_category', by='mock_category',
                         projection=gcrs.PlateCarree())
        finally:
            plt.close()

    def test_cartogram(self):
        try:
            gplt.cartogram(dataframe_gaussian_polys, scale='hue_var', facecolor='white')
            gplt.cartogram(dataframe_gaussian_polys, scale='hue_var', projection=gcrs.PlateCarree(), facecolor='white')

            gplt.cartogram(dataframe_gaussian_polys, scale='hue_var', legend_kwargs={'fancybox': False})
            gplt.cartogram(dataframe_gaussian_polys, scale='hue_var',
                           projection=gcrs.PlateCarree(), legend_kwargs={'fancybox': False})
        finally:
            plt.close()

    def test_sankey(self):
        gplt.sankey(path=list_paths, edgecolor='white')
        gplt.sankey(path=list_paths, projection=gcrs.PlateCarree(), edgecolor='white')

        gplt.sankey(path=list_paths, color='white')
        gplt.sankey(path=list_paths, projection=gcrs.PlateCarree(), color='white')

        gplt.sankey(path=list_paths, linewidth=1)
        gplt.sankey(path=list_paths, projection=gcrs.PlateCarree(), linewidth=1)

        gplt.sankey(path=list_paths, linestyle='--')
        gplt.sankey(path=list_paths, projection=gcrs.PlateCarree(), linestyle='--')
