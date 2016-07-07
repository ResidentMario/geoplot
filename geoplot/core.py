import folium
import geopandas as gpd
from geopandas import GeoDataFrame, GeoSeries
# import numpy as np
# import pandas as pd
# import functools
# import mplleaflet
import math


def _initialize_folium_layer(geom, padding=None):
    """
    Generates the folium layer for a plot.
    """
    # TODO: Something is wrong with this, see the notebook.
    print(geom.unary_union.bounds)
    x_min, y_min, x_max, y_max = geom.unary_union.bounds
    # center = (x_min + x_max) / 2, (y_min + y_max) / 2
    # print(center)
    map_layer = folium.Map()
    map_layer.fit_bounds([(x_min, y_min), (x_max, y_max)]) # , padding=padding)
    return map_layer


def point(geo, **kwargs):
    """
    Implements a point plot.
    :param geo:
    :param kwargs:
    :return:
    """
    map_layer = None
    if isinstance(geo, GeoSeries):
        map_layer = _initialize_folium_layer(geo)
        for geom in geo:
            folium.Marker([geom.y, geom.x]).add_to(map_layer)
    else: # GeoDataFrame
        geom_col = kwargs.pop('geometry', 'geometry')
        map_layer = _initialize_folium_layer(geo[geom_col])
    return map_layer