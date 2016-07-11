import folium
from folium.folium import Map
import geopandas as gpd
import pandas as pd
from pandas import DataFrame, Series
from geopandas import GeoDataFrame, GeoSeries
from shapely.geometry import Point
import matplotlib.pyplot as plt
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


def _initialize_folium_layer(geom, padding=None, scale=None,
                             tiles='OpenStreetMap',
                             attribution=None):
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
    # print(geom)
    # print(geom.unary_union)
    x_min, y_min, x_max, y_max = geom.unary_union.bounds

    # Handle padding.
    # Folium allows settings left and right bounds separately, and expects input in the form (n_px, n_px).
    # For example, folium.Map().fit_bounds([...], padding=(100, 100)) would ensure 100 pixels of boundary.
    # We will simplify the API by expecting a single value (some number n_px) and converting that.
    padding = [padding]*2

    # Folium has a control_scale parameter, however I think simply "scale" is much cleaner.
    map_layer = folium.Map(control_scale=scale, tiles=tiles, attr=attribution)
    map_layer.fit_bounds([(y_min, x_min), (y_max, x_max)], padding=padding)
    return map_layer


def point(data, padding=None, scale=False, radius=None, radial_func=None, tiles='OpenStreetMap'):
    # TODO: Decide if data in *args (as usual) or data in **kwargs (as in seaborn).
    """
    Implements a point plot.

    Parameters
    ----------
    data: GeoSeries or GeoDataFrame
        The geometry being plotted.
    padding: int or float
        The amount of padding to include in the plot, in pixels. This will cause geoplot to pick a Folium zoom level
        which results in at least this much space on any one side of the plot. Defaults to None.
    scale: bool
        Whether or not to display a map scale bar. Defaults to False.
    tiles: str
        A Leaflet-style URL of the form http://{s}.yourtiles.com/{z}/{x}/{y}.png which links to a known tilemap. See
        https://leaflet-extras.github.io/leaflet-providers/preview/ for a list of options. Defaults to OpenStreetMap
        if left unspecified.
    radius: iterable or int or float
        Controller for the radius of the displayed circles. If specified as an int or float, every circle will be
        this size. If specified as an iterable, will use that iterable's linearly normalized values,
        unless a different radial_func is specified.
    radial_func: function
        Optional parameter, may only be specified if a radius is specified as a str or an iterable (not an int or
        float). By default the point plot will plot a linearized radial length. A radial function can be specified
        to apply a different function to it instead. This is useful for cases in which linear radial length is not
        appropriate, for example when the parameter is highly imbalanced, in which case a log-linear radial map
        would be better.
    cmap:

    Returns
    -------
    A point map.
    """
    # Fetch the geometry column---the GeoSeries itself if one is passed, the requisite column if passed a GoDataFrame.
    # Before doing anything else, remove the geometries from the input containing NaN values, as this can cause the
    # process to fail at several key steps. GeoDataFrames do not handle missing values as well as I wish...
    geometries = data if isinstance(data, GeoSeries) else data._get_geometry()
    geometries = geometries[~geometries.is_empty & geometries.is_valid]

    # Initialize the map layer.
    map_layer = _initialize_folium_layer(data, padding=padding, scale=scale, tiles=tiles)

    # Set up the radii based on input.
    if isinstance(radius, int) or isinstance(radius, float):
        assert not radial_func, "A radial function does not make sense if the radius is specified as an int or a float."
        radii = [radius]*len(geometries)
    elif hasattr(radius, '__iter__'):  # radius is really a list of radii in this case
        assert len(radius) == len(geometries), "If an iterable is passed to radius it must be the same length as the " \
                                               "data."
        if not radial_func:
            radii = [datum/max(radius)*500 for datum in radius]
        else:
            radii = [radial_func(datum) for datum in radius]
    elif not radius:  # This is best placed here because when at the top input ndarrays raise ambiguity ValueErrors.
        # TODO: The Folium radius value-pass is the radius at maximum zoom. Figure out how to calculate a better one.
        assert not radial_func, "A radial function does not make sense if no radius is specified."
        radii = [500]*len(geometries)
    else:
        raise ValueError("Radius must be specified as an int, float, or iterable.")

    # Create markers and add them to the map.
    # Geometries with unspecified coordinates must be explicitly filtered out here.
    # This is because the addition to the map layer of a marker with NaN coordinates will cause Folium to silently
    # fail to plot any of the markers which come afterwards, whether their coordinates are valid or not.
    # cf. https://github.com/python-visualization/folium/issues/461
    markers = [folium.CircleMarker([geometry.y, geometry.x], radius=radii[i]) for i, geometry in enumerate(geometries)]
    for marker in markers:
        map_layer.add_children(marker)

    # Final result.
    return map_layer


def cluster(data, **kwargs):
    pass
    # TODO: https://github.com/python-visualization/folium/blob/master/examples/clustered_markers.ipynb