import folium
from folium.folium import Map
import geopandas as gpd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point
# import numpy as np
# import pandas as pd
# import functools
# import mplleaflet


# def __cast_as_geodataframe(data, x=None, y=None, coords=None, reversed=False):
#     """
#     Re-constitutes the data input as a GeoDataFrame. This is used to unify the plotting operations used throughout
#     the rest of the library.
#
#     Parameters
#     ----------
#     data: DataFrame, GeoDataFrame, or GeoSeries
#         The object being cast.
#
#     x: str
#         An x coordinate str. Used, if input is DataFrame and both x and y are specified, to recast into a GeoDataFrame.
#     y: str
#         A y coordinate str. Used, if input is DataFrame and both x and y are specified, to recast into a GeoDataFrame.
#     coords: str
#         A coordinate list str. Used, if input is either a Series OR DataFrame WITH coords specified, to recast into a
#         GeoDataFrame.
#     reversed: bool
#         By default this method expects coords input as (x, y) pairs (longitude-latitude order). If this is set to
#         true it will instead expect coords input as (y, x) pairs (historical latitude-longitude order).
#
#     Returns
#     -------
#     If input is a GeoDataFrame: simply returns the GeoDataFrame again.
#     If input is a GeoSeries: up-vets to a GeoDataFrame and returns it.
#     If input is a DataFrame: recasts into a GeoDataFrame using the additional input provided and returns that. If
#     not enough additional information is provided, or conflicting information is provided, raises an exception and
#     returns nothing.
#     If input is a Series: assumes that the Series contents are coordinate pairs and attempts to recast them into a
#     GeoDataFrame. If this fails, raises an exception and returns nothing.
#     If input is anything else: raises an exception and returns nothing.
#     """
#     if isinstance(data, GeoDataFrame):
#         return data
#     elif isinstance(data, GeoSeries):
#         return GeoDataFrame(data)
#     elif isinstance(data, Series):
#         if reversed:
#             return GeoDataFrame([Point(coord_pair[::-1]) for coord_pair in data], name=data.name)
#         else:
#             return GeoDataFrame([Point(coord_pair) for coord_pair in data], name=data.name)
#     elif isinstance(data, DataFrame):
#         if coords and not x and not y: # coords are specified, x and y are not.
#             if reversed:
#                 geometry = [Point(coord_pair[::-1]) for coord_pair in data[coords]]
#                 return GeoDataFrame(data[[col for col in data.columns if col != coords]], geometry=geometry)
#             else:
#                 geometry = [Point(coord_pair) for coord_pair in data[coords]]
#                 return GeoDataFrame(data[[col for col in data.columns if col != coords]], geometry=geometry)
#         elif (coords and x) or (coords and y): # coords are specified, one of or both of x and y also.
#             raise ValueError("Both (x, y) and coords input was provided. This is ambiguous.")
#         elif x and y: # x and y are specified, coords are not (as tested above).
#             geometry = [Point(coord_pair[::-1]) for coord_pair in data[coords]]
#             return GeoDataFrame()


def _initialize_folium_layer(geom, **kwargs):
    """
    Creates and returns a centered and padded Folium map for the given plot input.

    Parameters
    ----------
    geom: GeoSeries or GeoDataFrame
        The geometry being plotted.

    Returns
    -------
    A centered and padded Folium map.
    """
    # Calculate the geospatial envelope.
    # This will be passed to the Folium `fit_bounds` method in order to set the view.
    x_min, y_min, x_max, y_max = geom.unary_union.bounds
    # With regards to padding:
    # Folium allows settings left and right bounds separately, and expects input in the form (n_px, n_px).
    # For example, folium.Map().fit_bounds([...], padding=(100, 100)) would ensure 100 pixels of boundary.
    # We will simplify the API by expecting a single value (some number n_px) and converting that.
    padding = [kwargs.pop('padding', None)]*2
    # Folium has a control_scale parameter, however I think simply "scale" is much cleaner.
    scale = kwargs.pop('scale', None)
    map_layer = folium.folium.Map(**kwargs, control_scale=scale)
    map_layer.fit_bounds([(y_min, x_min), (y_max, x_max)], padding=padding)
    return map_layer


def point(data, **kwargs):
    """
    Implements a point plot.

    Parameters
    ----------
    geom: GeoSeries or GeoDataFrame
        The geometry being plotted.

    Returns
    -------
    A centered and padded Folium map.
    """
    map_layer = _initialize_folium_layer(data, padding=kwargs.pop('padding', None))
    geometries = data if isinstance(data, GeoSeries) else data._get_geometry()
    for geometry in geometries:
        map_layer.add_children(folium.Marker([geometry.y, geometry.x]))
    return map_layer