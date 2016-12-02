"""
This module defines the majority of geoplot functions, including all plot types.
"""

import geopandas as gpd
from geopandas.plotting import __pysal_choro, norm_cmap
# TODO: Clean up imports using import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import Normalize
import matplotlib.cm
from matplotlib.lines import Line2D
import numpy as np
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import warnings
from geoplot.quad import QuadTree
import shapely.geometry
# from matplotlib.patches import Rectangle
# from matplotlib.collections import PatchCollection
# import descartes
# from matplotlib.patches import Rectangle
# from matplotlib.collections import PatchCollection
# from scipy.spatial import KDTree
# from collections import defaultdict
# import pandas as pd
# from shapely import geometry


def pointplot(df,
              extent=None,
              hue=None,
              categorical=False, scheme=None, k=None, cmap='Set1', vmin=None, vmax=None,
              stock_image=False, coastlines=False, gridlines=False,
              projection=None,
              legend=False, legend_kwargs=None,
              figsize=(8, 6),
              **kwargs):
    """
    Generates an instance of a pointplot, the simplest kind of plot type available in this library. A pointplot is,
    at its core, simply a dot map, with each dot corresponding with a single geometric (x, y) point in the dataset.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        A geographic coordinate reference system projection. Must be an instance of an object in the ``geoplot.crs``
        module, e.g. ``geoplot.crs.PlateCarree()``. Refer to ``geoplot.crs`` for further object parameters.

        If this parameter is not specified this method will return an unprojected pure ``matplotlib`` version of the
        chart, as opposed to the ``cartopy`` based plot returned when a projection is present. This allows certain
        operations, like stacking ``geoplot`` plots with and amongst other plots, which are not possible when a
        projection is present.

        However, for the moment failing to specify a projection will raise a ``NotImplementedError``.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The data column whose entries are being discretely colorized. May be passed in any of a number of flexible
        formats. Defaults to None, in which case no colormap will be applied at all.
    categorical : boolean, optional
        Whether the inputted ``hue`` is already a categorical variable or not. Defaults to False. Ignored if ``hue``
        is set to None or not specified.
    scheme : None or {"quartiles"|"quantiles"|"equal_interval"|"fisher_jenks"} (?), optional
        The PySAL scheme which will be used to determine categorical bins for the ``hue`` choropleth. If ``hue`` is
        left unspecified or set to None this variable is ignored.
    k : int, optional
        If ``hue`` is specified and ``categorical`` is False, this number, set to 5 by default, will determine how
        many bins will exist in the output visualization. If ``hue`` is left unspecified or set to None this
        variable is ignored.
    cmap : matplotlib color, optional
        The string representation for a matplotlib colormap to be applied to this dataset. ``hue`` must be non-empty
        for a colormap to be applied at all, so this parameter is ignored otherwise.
    vmin : float, optional
        A strict floor on the value associated with the "bottom" of the colormap spectrum. Data column entries whose
        value is below this level will all be colored by the same threshold value. The value for this variable is
        meant to be inherited from the top-level variable of the same name.
    vmax : float, optional
        A strict ceiling on the value associated with the "top" of the colormap spectrum. Data column entries whose
        value is above this level will all be colored by the same threshold value. The value for this variable is
        meant to be inherited from the top-level variable of the same name.
    legend : boolean, optional
        Whether or not to include a legend in the output plot. This parameter will be ignored if ``hue`` is set to
        None or left unspecified.
    legend_kwargs : dict, optional
        Keword arguments to be passed to the ``matplotlib`` ``ax.legend`` method. For a list of possible arguments cf.
        http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend.
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    stock_image : boolean, optional
        Whether or not to overlay the low-resolution Natural Earth world map.
    coastlines : boolean, optional
        Whether or not to overlay the low-resolution Natural Earth coastlines.
    gridlines : boolean, optional
        Whether or not to overlay cartopy's computed latitude-longitude gridlines.
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is set to None (default) this method will calculate its own cartographic display region. If
        an extrema tuple is passed---useful if you want to focus on a particular area, for example, or exclude certain
        outliers---that input will be used instead.
    kwargs: dict, optional
        Keyword arguments to be passed to the ``ax.scatter`` method doing the plotting. For a list of possible
        arguments cf. http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter

    Returns
    -------
    None
        Terminates by calling ``plt.show()``.
    """
    # Initialize the figure.
    fig = plt.figure(figsize=figsize)

    # If a hue parameter is specified and is a string, convert it to a reference to its column. This puts us on a
    # level playing field with cases when hue is specified as an explicit iterable. If hue is None, do nothing.
    if isinstance(hue, str):
        hue = df[hue]

    # Validate bucketing.
    categorical, k, scheme = _validate_buckets(categorical, k, scheme)

    # TODO: Work this out.
    # # If we are not handed a projection we are in the PateCarree projection. In that case we can return a
    # # `matplotlib` plot directly, which has the advantage of being native to e.g. mplleaflet.
    if not projection:
        raise NotImplementedError
        # xs = np.array([p.x for p in df.geometry])
        # ys = np.array([p.y for p in df.geometry])
        # return plt.scatter(xs, ys)

    # Properly set up the projection.
    projection = projection.load(df, {
        'central_longitude': lambda df: np.mean(np.array([p.x for p in df.geometry.centroid])),
        'central_latitude': lambda df: np.mean(np.array([p.y for p in df.geometry.centroid]))
    })

    # Set up the axis. Note that even though the method signature is from matplotlib, after this operation ax is a
    # cartopy.mpl.geoaxes.GeoAxesSubplot object! This is a subclass of a matplotlib Axes class but not directly
    # compatible with one, so it means that this axis cannot, for example, be plotted using mplleaflet.
    ax = plt.subplot(111, projection=projection)

    # Set extent. In order to prevent points from being occluded, we set it to be a little bit larger than the values
    # in the plot themselves. This is done within the data itself because the underlying plot appears not to respect
    # commands like e.g. ax.margin(0.05), which would have a similar effect.
    # Currently 5% of the plot area is reserved for padding.
    xs = np.array([p.x for p in df.geometry])
    ys = np.array([p.y for p in df.geometry])

    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())

    # Set optional parameters.
    _set_optional_parameters(ax, stock_image, coastlines, gridlines)

    # Clean up patches.
    _lay_out_axes(ax)

    # Set up the colormap. This code is largely taken from geoplot's choropleth facilities, cf.
    # https://github.com/geopandas/geopandas/blob/master/geopandas/plotting.py#L253
    # If a scheme is provided we compute a distribution for the given data. If one is not provided we assume that the
    # input data is categorical.
    if hue is not None:
        cmap, categories, values = _discrete_colorize(categorical, hue, scheme, k, cmap, vmin, vmax)
        colors = [cmap.to_rgba(v) for v in values]

        if legend:
            _paint_legend(ax, categories, cmap, legend_kwargs)
    else:
        colors = 'steelblue'

    # Draw. Notice that this scatter method's signature is attached to the axis instead of to the overall plot. This
    # is again because the axis is a special cartopy object.
    ax.scatter(xs, ys, transform=ccrs.PlateCarree(), c=colors, **kwargs)
    plt.show()


def choropleth(df,
               projection=None,
               hue=None,
               scheme=None, k=None, cmap='Set1', categorical=False, vmin=None, vmax=None,
               legend=False, legend_kwargs=None,
               stock_image=False, coastlines=False, gridlines=False,
               extent=None,
               figsize=(8, 6),
               **kwargs):
    """
    Generates an instance of a choropleth, a simple aggregation plot based on geometry properties and a mainstay of
    the geospatial data science field.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        A geographic coordinate reference system projection. Must be an instance of an object in the ``geoplot.crs``
        module, e.g. ``geoplot.crs.PlateCarree()``. Refer to ``geoplot.crs`` for further object parameters.

        If this parameter is not specified this method will return an unprojected pure ``matplotlib`` version of the
        chart, as opposed to the ``cartopy`` based plot returned when a projection is present. This allows certain
        operations, like stacking ``geoplot`` plots with and amongst other plots, which are not possible when a
        projection is present.

        However, for the moment failing to specify a projection will raise a ``NotImplementedError``.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The data column whose entries are being discretely colorized. May be passed in any of a number of flexible
        formats. Defaults to None, in which case no colormap will be applied at all.
    categorical : boolean, optional
        Whether the inputted ``hue`` is already a categorical variable or not. Defaults to False. Ignored if ``hue``
        is set to None or not specified.
    scheme : None or {"quartiles"|"quantiles"|"equal_interval"|"fisher_jenks"} (?), optional
        The PySAL scheme which will be used to determine categorical bins for the ``hue`` choropleth. If ``hue`` is
        left unspecified or set to None this variable is ignored.
    k : int, optional
        If ``hue`` is specified and ``categorical`` is False, this number, set to 5 by default, will determine how
        many bins will exist in the output visualization. If ``hue`` is left unspecified or set to None this
        variable is ignored.
    cmap : matplotlib color, optional
        The string representation for a matplotlib colormap to be applied to this dataset. ``hue`` must be non-empty
        for a colormap to be applied at all, so this parameter is ignored otherwise.
    vmin : float, optional
        A strict floor on the value associated with the "bottom" of the colormap spectrum. Data column entries whose
        value is below this level will all be colored by the same threshold value. The value for this variable is
        meant to be inherited from the top-level variable of the same name.
    vmax : float, optional
        A strict ceiling on the value associated with the "top" of the colormap spectrum. Data column entries whose
        value is above this level will all be colored by the same threshold value. The value for this variable is
        meant to be inherited from the top-level variable of the same name.
    legend : boolean, optional
        Whether or not to include a legend in the output plot. This parameter will be ignored if ``hue`` is set to
        None or left unspecified.
    legend_kwargs : dict, optional
        Keword arguments to be passed to the ``matplotlib`` ``ax.legend`` method. For a list of possible arguments cf.
        http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend.
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    stock_image : boolean, optional
        Whether or not to overlay the low-resolution Natural Earth world map.
    coastlines : boolean, optional
        Whether or not to overlay the low-resolution Natural Earth coastlines.
    gridlines : boolean, optional
        Whether or not to overlay cartopy's computed latitude-longitude gridlines.
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is set to None (default) this method will calculate its own cartographic display region. If
        an extrema tuple is passed---useful if you want to focus on a particular area, for example, or exclude certain
        outliers---that input will be used instead.
    kwargs: dict, optional
        Keyword arguments to be passed to the ``ax.scatter`` method doing the plotting. For a list of possible
        arguments cf. http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter

    Returns
    -------
    None
        Terminates by calling ``plt.show()``.
    """

    # Format the data to be displayed for input.
    hue = _validate_hue(df, hue)

    # Validate bucketing.
    categorical, k, scheme = _validate_buckets(categorical, k, scheme)

    # Initialize the figure.
    fig = plt.figure(figsize=figsize)

    # If we are not handed a projection we are in the PateCarree projection. In that case we can return a
    # `matplotlib` plot directly, which has the advantage of being native to e.g. mplleaflet.
    # TODO: Implement this.
    if not projection:
        raise NotImplementedError

    projection = projection.load(df, {
        'central_longitude': lambda df: np.mean(np.array([p.x for p in df.geometry.centroid])),
        'central_latitude': lambda df: np.mean(np.array([p.y for p in df.geometry.centroid]))
    })

    # Set up the axis. Note that even though the method signature is from matplotlib, after this operation ax is a
    # cartopy.mpl.geoaxes.GeoAxesSubplot object! This is a subclass of a matplotlib Axes class but not directly
    # compatible with one, so it means that this axis cannot, for example, be plotted using mplleaflet.
    ax = plt.subplot(111, projection=projection)

    # Set extent.
    x_min_coord, x_max_coord, y_min_coord, y_max_coord = _get_envelopes_min_maxes(df.geometry.envelope.exterior)
    if extent:
        ax.set_extent(extent, crs=ccrs.PlateCarree())
    else:
        ax.set_extent((x_min_coord, x_max_coord, y_min_coord, y_max_coord), crs=ccrs.PlateCarree())

    # Set optional parameters.
    _set_optional_parameters(ax, stock_image, coastlines, gridlines)

    # Generate colormaps.
    cmap, categories, values = _discrete_colorize(categorical, hue, scheme, k, cmap, vmin, vmax)

    # Clean up patches.
    _lay_out_axes(ax)

    if legend:
        _paint_legend(ax, categories, cmap, legend_kwargs)

    # Finally we draw the features.
    for cat, geom in zip(values, df.geometry):
        features = ShapelyFeature([geom], ccrs.PlateCarree())
        ax.add_feature(features, facecolor=cmap.to_rgba(cat), **kwargs)
    plt.show()


def aggplot(df,
            projection=None,
            by=None,
            threshold=5,
            hue=None, cmap='Set1', vmin=None, vmax=None,
            agg=np.mean,
            # legend=False, legend_kwargs=None,
            extent=None,
            figsize=(8, 6),
            **kwargs):
    """
    Generates an instance of an aggregate plot, a minimum-expectations summary plot type which handles mixes of
    geometry types and missing aggregate geometry data.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        A geographic coordinate reference system projection. Must be an instance of an object in the ``geoplot.crs``
        module, e.g. ``geoplot.crs.PlateCarree()``. Refer to ``geoplot.crs`` for further object parameters.

        If this parameter is not specified this method will return an unprojected pure ``matplotlib`` version of the
        chart, as opposed to the ``cartopy`` based plot returned when a projection is present. This allows certain
        operations, like stacking ``geoplot`` plots with and amongst other plots, which are not possible when a
        projection is present.

        However, for the moment failing to specify a projection will raise a ``NotImplementedError``.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The data column whose entries are being discretely colorized. May be passed in any of a number of flexible
        formats. Defaults to None, in which case no colormap will be applied at all.
    categorical : boolean, optional
        Whether the inputted ``hue`` is already a categorical variable or not. Defaults to False. Ignored if ``hue``
        is set to None or not specified.
    scheme : None or {"quartiles"|"quantiles"|"equal_interval"|"fisher_jenks"} (?), optional
        The PySAL scheme which will be used to determine categorical bins for the ``hue`` choropleth. If ``hue`` is
        left unspecified or set to None this variable is ignored.
    k : int, optional
        If ``hue`` is specified and ``categorical`` is False, this number, set to 5 by default, will determine how
        many bins will exist in the output visualization. If ``hue`` is left unspecified or set to None this
        variable is ignored.
    cmap : matplotlib color, optional
        The string representation for a matplotlib colormap to be applied to this dataset. ``hue`` must be non-empty
        for a colormap to be applied at all, so this parameter is ignored otherwise.
    vmin : float, optional
        A strict floor on the value associated with the "bottom" of the colormap spectrum. Data column entries whose
        value is below this level will all be colored by the same threshold value. The value for this variable is
        meant to be inherited from the top-level variable of the same name.
    vmax : float, optional
        A strict ceiling on the value associated with the "top" of the colormap spectrum. Data column entries whose
        value is above this level will all be colored by the same threshold value. The value for this variable is
        meant to be inherited from the top-level variable of the same name.
    legend : boolean, optional
        Whether or not to include a legend in the output plot. This parameter will be ignored if ``hue`` is set to
        None or left unspecified.
    legend_kwargs : dict, optional
        Keword arguments to be passed to the ``matplotlib`` ``ax.legend`` method. For a list of possible arguments cf.
        http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend.
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    gridlines : boolean, optional
        Whether or not to overlay cartopy's computed latitude-longitude gridlines.
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is set to None (default) this method will calculate its own cartographic display region. If
        an extrema tuple is passed---useful if you want to focus on a particular area, for example, or exclude certain
        outliers---that input will be used instead.
    kwargs: dict, optional
        Keyword arguments to be passed to the ``ax.scatter`` method doing the plotting. For a list of possible
        arguments cf. http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter

    Returns
    -------
    None
        Terminates by calling ``plt.show()``.
    """
    # TODO: Parameters are incomplete, so docstring is likewise.

    fig = plt.plot(figsize=figsize)

    # TODO: Implement this.
    if not projection:
        raise NotImplementedError

    projection = projection.load(df, {
        'central_longitude': lambda df: np.mean(np.array([p.x for p in df.geometry.centroid])),
        'central_latitude': lambda df: np.mean(np.array([p.y for p in df.geometry.centroid]))
    })

    ax = plt.subplot(111, projection=projection)

    # # Generate colormaps.
    # cmap, categories, values = _colorize(categorical, hue, scheme, k, cmap, vmin, vmax)

    # Clean up patches.
    _lay_out_axes(ax)

    # Format hue and generate a colormap
    hue_col = hue
    values = _validate_hue(df, hue)
    cmap = _continuous_colormap(values, cmap, vmin, vmax)

    if by:
        pass
    else:
        # Generate a quadtree and partition it to generate.
        quad = QuadTree(df)
        bxmin, bxmax, bymin, bymax = quad.bounds
        partitions = quad.partition(threshold)
        for p in partitions:
            xmin, xmax, ymin, ymax = p.bounds
            rect = shapely.geometry.Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
            feature = ShapelyFeature([rect], ccrs.PlateCarree())
            color = cmap.to_rgba(agg(p.data[hue_col]))
            ax.add_feature(feature, facecolor=color, **kwargs)
            # TODO: The code snippet for matplotlib alone is below.
            # ax.add_artist(Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, facecolor='lightgray'))
            # Note: patches.append(...); ax.add_collection(PatchCollection(patches)) will not work.
            # cf. http://stackoverflow.com/questions/10550477/how-do-i-set-color-to-rectangle-in-matplotlib
        if extent:
            ax.set_extent(extent, crs=ccrs.PlateCarree())
        else:
            ax.set_extent((bxmin, bxmax, bymin, bymax), crs=ccrs.PlateCarree())
    plt.show()


##################
# HELPER METHODS #
##################


def _get_envelopes_min_maxes(envelopes):
    """
    Returns the extrema of the inputted polygonal envelopes. Used for setting chart extent where appropriate. Note
    tha the ``Quadtree.bounds`` object property serves a similar role.

    Parameters
    ----------
    envelopes : GeoSeries
        The envelopes of the given geometries, as would be returned by e.g. ``data.geometry.envelope``.

    Returns
    -------
    (xmin, xmax, ymin, ymax) : tuple
        The data extrema.

    """
    xmin = np.min(envelopes.map(lambda linearring: np.min([linearring.coords[1][0],
                                                          linearring.coords[2][0],
                                                          linearring.coords[3][0],
                                                          linearring.coords[4][0]])))
    xmax = np.max(envelopes.map(lambda linearring: np.max([linearring.coords[1][0],
                                                          linearring.coords[2][0],
                                                          linearring.coords[3][0],
                                                          linearring.coords[4][0]])))
    ymin = np.min(envelopes.map(lambda linearring: np.min([linearring.coords[1][1],
                                                           linearring.coords[2][1],
                                                           linearring.coords[3][1],
                                                           linearring.coords[4][1]])))
    ymax = np.max(envelopes.map(lambda linearring: np.max([linearring.coords[1][1],
                                                           linearring.coords[2][1],
                                                           linearring.coords[3][1],
                                                           linearring.coords[4][1]])))
    return xmin, xmax, ymin, ymax


def _get_envelopes_centroid(envelopes):
    """
    Returns the centroid of an inputted geometry column. Not currently in use, as this is now handled by this
    library's CRS wrapper directly. Light wrapper over ``_get_envelopes_min_maxes``.

    Parameters
    ----------
    envelopes : GeoSeries
        The envelopes of the given geometries, as would be returned by e.g. ``data.geometry.envelope``.

    Returns
    -------
    (mean_x, mean_y) : tuple
        The data centroid.
    """
    xmin, xmax, ymin, ymax = _get_envelopes_min_maxes(envelopes)
    return np.mean(xmin, xmax), np.mean(ymin, ymax)


def _lay_out_axes(ax):
    """
    Cartopy enables a a transparent background patch and an "outline" patch by default. This short method simply
    hides these extraneous visual features.

    Parameters
    ----------
    ax : matplotlib.Axes instance
        The ``matplotlib.Axes`` instance being manipulated.

    Returns
    -------
    None
    """
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)


def _validate_hue(df, hue):
    """
    The top-level ``hue`` parameter present in most plot types accepts a variety of input types. This method
    condenses this variety into a single preferred format---an iterable---which is expected by all submethods working
    with the data downstream of it.

    Parameters
    ----------
    df : GeoDataFrame
        The full data input, from which standardized ``hue`` information may need to be extracted.
    hue : Series, GeoSeries, iterable, str
        The data column whose entries are being discretely colorized, as (loosely) passed by the top-level ``hue``
        variable.

    Returns
    -------
    hue : iterable
        The ``hue`` parameter input as an iterable.
    """
    if not hue:
        nongeom = set(df.columns) - {df.geometry.name}
        if len(nongeom) > 1:
            raise ValueError("Ambiguous input: no 'hue' parameter was specified and the inputted DataFrame has more "
                             "than one column of data.")
        else:
            hue = df[list(nongeom)[0]]
            return hue
    elif isinstance(hue, str):
        hue = df[hue]
        return hue


def _continuous_colormap(hue, cmap, vmin, vmax):
    """
    Creates a continuous colormap.

    Parameters
    ----------
    hue : iterable
        The data column whose entries are being discretely colorized. Note that although top-level plotter ``hue``
        parameters ingest many argument signatures, not just iterables, they are all preprocessed to standardized
        iterables before this method is called.
    cmap : ``matplotlib.cm`` instance
        The `matplotlib` colormap instance which will be used to colorize the geometries.
    vmin : float
        A strict floor on the value associated with the "bottom" of the colormap spectrum. Data column entries whose
        value is below this level will all be colored by the same threshold value. The value for this variable is
        meant to be inherited from the top-level variable of the same name.
    vmax : float
        A strict ceiling on the value associated with the "top" of the colormap spectrum. Data column entries whose
        value is above this level will all be colored by the same threshold value. The value for this variable is
        meant to be inherited from the top-level variable of the same name.

    Returns
    -------
    cmap : ``mpl.cm.ScalarMappable`` instance
        A normalized scalar version of the input ``cmap`` which has been fitted to the data and inputs.
    """
    mn = min(hue) if vmin is None else vmin
    mx = max(hue) if vmax is None else vmax
    norm = Normalize(vmin=mn, vmax=mx)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def _discrete_colorize(categorical, hue, scheme, k, cmap, vmin, vmax):
    """
    Creates a discrete colormap, either using an already-categorical data variable or by bucketing a non-categorical
    ordinal one. If a scheme is provided we compute a distribution for the given data. If one is not provided we
    assume that the input data is categorical.

    This code is based on the ``geopandas`` choropleth facilities, cf.
    https://github.com/geopandas/geopandas/blob/master/geopandas/plotting.py#L253

    Parameters
    ----------
    categorical : boolean
        Whether or not the input variable is already categorical.
    hue : iterable
        The data column whose entries are being discretely colorized. Note that although top-level plotter ``hue``
        parameters ingest many argument signatures, not just iterables, they are all preprocessed to standardized
        iterables before this method is called.
    scheme : str
        The PySAL binning scheme to be used for splitting data values (or rather, the the string representation
        thereof).
    k : int
        The number of bins which will be used. This parameter will be ignored if ``categorical`` is True. The default
        value should be 5---this should be set before this method is called.
    cmap : ``matplotlib.cm`` instance
        The `matplotlib` colormap instance which will be used to colorize the geometries. This colormap
        determines the spectrum; our algorithm determines the cuts.
    vmin : float
        A strict floor on the value associated with the "bottom" of the colormap spectrum. Data column entries whose
        value is below this level will all be colored by the same threshold value. The value for this variable is
        meant to be inherited from the top-level variable of the same name.
    vmax : float
        A strict cealing on the value associated with the "bottom" of the colormap spectrum. Data column entries whose
        value is above this level will all be colored by the same threshold value. The value for this variable is
        meant to be inherited from the top-level variable of the same name.

    Returns
    -------
    (cmap, categories, values) : tuple
        A tuple meant for assignment containing the values for various properties set by this method call.
    """
    if not categorical:
        binning = __pysal_choro(hue, scheme, k=k)
        values = binning.yb
        binedges = [binning.yb.min()] + binning.bins.tolist()
        categories = ['{0:.2f} - {1:.2f}'.format(binedges[i], binedges[i + 1])
                      for i in range(len(binedges) - 1)]
    else:
        categories = np.unique(hue)
        if len(categories) > 10:
            warnings.warn("Generating a choropleth using a categorical column with over 10 individual categories. "
                          "This is not recommended!")
        value_map = {v: i for i, v in enumerate(categories)}
        values = [value_map[d] for d in hue]
    cmap = norm_cmap(values, cmap, Normalize, matplotlib.cm, vmin=vmin, vmax=vmax)
    return cmap, categories, values


def _paint_legend(ax, categories, cmap, legend_kwargs):
    """
    Creates a legend and attaches it to the axis. Meant to be used when a ``legend=True`` parameter is passed.

    Parameters
    ----------
    ax : matplotlib.Axes instance
        The ``matplotlib.Axes`` instance on which a legend is being painted.
    categories : list
        A list of categories being plotted. May be either a list of int types or a list of unique entities in the
        data column (e.g. as generated via ``numpy.unique(data)``. This parameter is meant to be the same as that
        returned by the ``_discrete_colorize`` method.
    cmap : ``matplotlib.cm`` instance
        The `matplotlib` colormap instance which will be used to colorize the legend entries. This should be the
        same one used for colorizing the plot's geometries.
    legend_kwargs : dict
        Keyword arguments which will be passed to the matplotlib legend instance on initialization. This parameter
        is provided to allow fine-tuning of legend placement at the top level of a plot method, as legends are very
        finicky.

    Returns
    -------
    None.
    """
    patches = []
    for value, cat in enumerate(categories):
        patches.append(Line2D([0], [0], linestyle="none",
                              marker="o",
                              markersize=10, markerfacecolor=cmap.to_rgba(value)))
    # I can't initialize legend_kwargs as an empty dict() by default because of Python's argument mutability quirks.
    # cf. http://docs.python-guide.org/en/latest/writing/gotchas/. Instead my default argument is None,
    # but that doesn't unpack correctly, necessitating setting and passing an empty dict here. Awkward...
    if not legend_kwargs: legend_kwargs = dict()
    ax.legend(patches, categories, numpoints=1, fancybox=True, **legend_kwargs)


def _set_optional_parameters(ax, stock_image, coastlines, gridlines):
    """
    A utility method which handles in-place various ``cartopy`` onboards available within certain plot types.

    Parameters
    ----------
    ax : matplotlib.Axes instance
        The ``matplotlib.Axes`` instance whose properties are being toggled.
    stock_image : boolean
        Whether or not to overlay the low-resolution Natural Earth world map.
    coastlines : boolean
        Whether or not to overlay the low-resolution Natural Earth coastlines.
    gridlines : boolean
        Whether or not to overlay cartopy's computed latitude-longitude gridlines. Note: labelling is also possible,
        but in the current state of development of cartopy is only available for the PlateCarree and Mercator
        projections, so it has not yet been enabled here.

    Returns
    -------
    None
    """
    if stock_image:
        ax.stock_img()
    if coastlines:
        ax.coastlines()
    if gridlines:
        ax.gridlines()


def _validate_buckets(categorical, k, scheme):
    """
    This method validates that the hue parameter is correctly specified. Valid inputs are:

        1. Both k and scheme are specified. In that case the user wants us to handle binning the data into k buckets
           ourselves, using the stated algorithm. We issue a warning if the specified k is greater than 10.
        2. k is left unspecified and scheme is specified. In that case the user wants us to handle binning the data
           into some default (k=5) number of buckets, using the stated algorithm.
        3. Both k and scheme are left unspecified. In that case the user wants us bucket the data variable using some
           default algorithm (Quantiles) into some default number of buckets (5).
        4. k is specified, but scheme is not. We choose to interpret this as meaning that the user wants us to handle
           bucketing the data into k buckets using the default (Quantiles) bucketing algorithm.
        5. categorical is True, and both k and scheme are False or left unspecified. In that case we do categorical.
        Invalid inputs are:
        6. categorical is True, and one of k or scheme are also specified. In this case we raise a ValueError as this
           input makes no sense.

    Parameters
    ----------
    categorical : boolean
        Whether or not the data values given in ``hue`` are already a categorical variable.

    k : int
        The number of categories to use. This variable has no effect if ``categorical`` is True, and will be set to 5
        by default if it is False and not already given.

    scheme : str
        The PySAL scheme that the variable will be categorized according to (or rather, a string representation
        thereof).

    Returns
    -------
    (categorical, k, scheme) : tuple
        A possibly modified input tuple meant for reassignment in place.
    """
    if categorical and (k or scheme):
        raise ValueError("Invalid input: categorical cannot be specified as True simultaneously with scheme or k "
                         "parameters")
    if not k:
        k = 5
    if k > 10:
        warnings.warn("Generating a choropleth using a categorical column with over 10 individual categories. "
                      "This is not recommended!")
    if not scheme:
        scheme = 'Quantiles'  # This trips it correctly later.
    return categorical, k, scheme


def __indices_inside(df, window):
    """
    Returns the indices of columns in a ``geopandas`` object located within a certain rectangular window. This helper
    method is not currently used, see the quad package instead.

    Parameters
    ----------
    df : GeoSeries or GeoDataFrame
        The ``geopandas`` object containing a ``geometry`` of interest.
    window : tuple
        The ``(min_x, max_x, min_y, max_y)`` rectangular lookup coordinates from which points will be returned.

    Returns
    -------
    The indices within `df` of points within `window`.
    """
    min_x, max_x, min_y, max_y = window
    points = df.geometry.centroid
    is_in = points.map(lambda point: (min_x < point.x < max_x) & (min_y < point.y < max_y))
    indices = is_in.values.nonzero()[0]
    return indices
