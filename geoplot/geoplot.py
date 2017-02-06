"""
This module defines the majority of geoplot functions, including all plot types.
"""

import geopandas as gpd
from geopandas.plotting import __pysal_choro, norm_cmap
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import warnings
from geoplot.quad import QuadTree
import shapely.geometry
import pandas as pd
import descartes


def pointplot(df, projection=None,
              hue=None, categorical=False, scheme=None, k=5, cmap='Set1', vmin=None, vmax=None,
              scale=None, limits=(0.5, 2), scale_func=None,
              legend=False, legend_values=None, legend_labels=None, legend_kwargs=None, legend_var=None,
              figsize=(8, 6), extent=None, ax=None, **kwargs):
    """
    A geospatial scatter plot. The simplest useful plot type available.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        A geographic projection. Must be an instance of an object in the ``geoplot.crs`` module,
        e.g. ``geoplot.crs.PlateCarree()``. This parameter is optional: if left unspecified, a pure unprojected
        ``matplotlib`` object will be returned. For more information refer to the tutorial page on `projections
        <http://localhost:63342/geoplot/docs/_build/html/tutorial/projections.html>`_.
    hue : None, Series, GeoSeries, iterable, or str, optional
        A data column whose values are to be colorized. Defaults to None, in which case no colormap will be applied.
    categorical : boolean, optional
        Specify this variable to be ``True`` if ``hue`` points to a categorical variable. Defaults to False. Ignored
        if ``hue`` is set to None or not specified.
    scheme : None or {"Quantiles"|"Equal_interval"|"Fisher_Jenks"}, optional
        The scheme which will be used to determine categorical bins for the ``hue`` choropleth. If ``hue`` is
        left unspecified or set to None this variable is ignored.
    k : int or None, optional
        If ``hue`` is specified and ``categorical`` is False, this number, set to 5 by default, will determine how
        many bins will exist in the output visualization. If ``hue`` is specified and this variable is set to
        ``None``, a continuous colormap will be used. If ``hue`` is left unspecified or set to None this variable is
        ignored.
    cmap : matplotlib color, optional
        The matplotlib colormap to be applied to this dataset (`ref
        <http://matplotlib.org/examples/color/colormaps_reference.html>`_). This parameter is ignored if ``hue`` is not
        specified.
    vmin : float, optional
        The value that "bottoms out" the colormap. Data column entries whose value is below this level will be
        colored the same threshold value. Defaults to the minimum value in the dataset.
    vmax : float, optional
        The value that "tops out" the colormap. Data column entries whose value is above this level will be
        colored the same threshold value. Defaults to the maximum value in the dataset.
    scale : str or iterable, optional
        A data column whose values will be used to scale the points. Defaults to None, in which case no scaling will be
        applied.
    limits : (min, max) tuple, optional
        The minimum and maximum limits against which the shape will be scaled. Ignored if ``scale`` is not specified.
    scale_func : ufunc, optional
        The function used to scale point sizes. This should be a factory function of two 
        variables, the minimum and maximum values in the dataset, which returns a scaling function which will be 
        applied to the rest of the data. Defaults to a linear scale. A demo is available in the `example 
        gallery <examples/usa-city-elevations.html>`_.
    legend : boolean, optional
        Whether or not to include a legend in the output plot. This parameter will not work if neither ``hue`` nor
        ``scale`` is unspecified.
    legend_values : list, optional
        Equal intervals will be used for the "points" in the legend by default. However, particularly if your scale
        is non-linear, oftentimes this isn't what you want. If this variable is provided as well, the values
        included in the input will be used by the legend instead.
    legend_labels : list, optional
        If a legend is specified, this parameter can be used to control what names will be attached to the values.
    legend_var : "hue" or "scale", optional
        The name of the visual variable for which a legend will be displayed. Does nothing if ``legend`` is False
        or multiple variables aren't used simultaneously.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.pyplot.legend`` instance (`ref
        <http://matplotlib.org/users/legend_guide.html>`_).
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is unset ``geoplot`` will calculate the plot limits. If an extrema tuple is passed,
        that input will be used instead.
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        A ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot`` instance onto which this plot
        will be graphed. If this parameter is left undefined a new axis will be created and used instead.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.pyplot.scatter`` instance (`ref
        <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter>`_).

    Returns
    -------
    AxesSubplot or GeoAxesSubplot instance
        The axis object with the plot on it.



    Examples
    --------

    The ``pointplot`` is a simple `geospatial scatter plot <https://en.wikipedia.org/wiki/Scatter_plot>`_,
    with each point corresponding with one observation in your dataset. The expected input is a ``geopandas``
    ``GeoDataFrame`` with geometries consisting of ``shapely.geometry.Point`` entities. The simplest possible plot
    can be made by specifying the input data (and, optionally, a `projection <./tutorial/projections.html>`_).

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        gplt.pointplot(points)

    .. image:: ../figures/pointplot/pointplot-initial.png


    The ``hue`` parameter accepts a data column and applies a colormap to the output.

    .. code-block:: python

        gplt.pointplot(cities, projection=gcrs.AlbersEqualArea(), hue='ELEV_IN_FT')

    .. image:: ../figures/pointplot/pointplot-hue.png

    The ``legend`` parameter toggles a legend.

    .. code-block:: python

        gplt.pointplot(cities, projection=gcrs.AlbersEqualArea(), hue='ELEV_IN_FT', legend=True)

    .. image:: ../figures/pointplot/pointplot-legend.png

    ``legend_labels`` specifies custom legend labels.

    .. code-block:: python

        gplt.pointplot(cities, projection=gcrs.AlbersEqualArea(), hue='ELEV_IN_FT',
                       legend=True, legend_labels=list('ABCDE'))

    .. image:: ../figures/pointplot/pointplot-legend-labels.png

    ``pointplot`` will default to binning the observations in the given data column into five ordinal classes
    containing equal numbers of observations - a `quantile <https://en.wikipedia.org/wiki/Quantile>`_ scheme. An
    alternative binning scheme can be specified using the ``scheme`` parameter. Valid options are ``Quantile``,
    ``Equal_interval`` (bins will be of equal sizes but contain different numbers of observations),
    and ``Fisher_jenks`` (an intermediate between the two).

    .. code-block:: python

        gplt.pointplot(cities, projection=gcrs.AlbersEqualArea(), hue='ELEV_IN_FT',
                       legend=True, scheme='equal_interval')

    .. image:: ../figures/pointplot/pointplot-scheme.png

    If the variable of interest is already `categorical
    <http://pandas.pydata.org/pandas-docs/stable/categorical.html>`_, specify ``categorical=True`` to
    use the labels in your dataset directly.

    .. code-block:: python

        gplt.pointplot(collisions, projection=gcrs.AlbersEqualArea(), hue='BOROUGH',
                       legend=True, categorical=True)

    .. image:: ../figures/pointplot/pointplot-categorical.png

    Keyword arguments can be passed to the legend using the ``legend_kwargs`` argument. These arguments will be
    passed to the underlying ``matplotlib.legend.Legend`` instance (`ref
    <http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend>`_). The ``loc`` and ``bbox_to_anchor``
    parameters are particularly useful for positioning the legend.

    .. code-block:: python

        gplt.pointplot(collisions, projection=gcrs.AlbersEqualArea(), hue='BOROUGH',
                       categorical=True, legend=True, legend_kwargs={'loc': 'upper left'})

    .. image:: ../figures/pointplot/pointplot-legend-kwargs.png

    Additional arguments will be interpreted as keyword arguments to the underlying ``matplotlib.pyplot.scatter``
    instance (`ref <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter>`_).

    .. code-block:: python

        gplt.pointplot(collisions[collisions['BOROUGH'].notnull()], projection=gcrs.AlbersEqualArea(),
                       hue='BOROUGH', categorical=True,
                       legend=True, legend_kwargs={'loc': 'upper left'},
                       edgecolor='white', linewidth=0.5)

    .. image:: ../figures/pointplot/pointplot-kwargs.png

    Change the number of bins by specifying an alternative ``k`` value.

    .. code-block:: python

        gplt.pointplot(data, projection=gcrs.AlbersEqualArea(),
                       hue='var', k=8,
                       edgecolor='white', linewidth=0.5,
                       legend=True, legend_kwargs={'bbox_to_anchor': (1.25, 1.0)})

    .. image:: ../figures/pointplot/pointplot-k.png

    Adjust the `colormap <http://matplotlib.org/examples/color/colormaps_reference.html>`_ to any
    colormap recognizable to ``matplotlib`` using the ``cmap`` parameter.

    .. code-block:: python

        gplt.pointplot(data, projection=gcrs.AlbersEqualArea(),
               hue='var', cmap='inferno', k=8,
               edgecolor='white', linewidth=0.5,
               legend=True, legend_kwargs={'bbox_to_anchor': (1.25, 1.0)})


    .. image:: ../figures/pointplot/pointplot-cmap.png

    To use a continuous colormap, explicitly specify ``k=None``. Note that if ``legend=True``,
    a ``matplotlib.colorbar`` legend will be used (`ref <http://matplotlib.org/api/colorbar_api.html>`_).

    .. code-block:: python

        gplt.pointplot(data, projection=gcrs.AlbersEqualArea(),
               hue='var', cmap='inferno', k=None,
               edgecolor='white', linewidth=0.5,
               legend=True, legend_kwargs={'bbox_to_anchor': (1.25, 1.0)})

    .. image:: ../figures/pointplot/pointplot-k-None.png

    ``scale`` provides an alternative or additional visual variable.

    .. code-block:: python

        gplt.pointplot(collisions, projection=gcrs.AlbersEqualArea(),
                       scale='NUMBER OF PERSONS INJURED',
                       legend=True, legend_kwargs={'loc': 'upper left'})

    .. image:: ../figures/pointplot/pointplot-scale.png

    The limits can be adjusted to fit your data using the ``limits`` parameter.

    .. code-block:: python

        gplt.pointplot(collisions, projection=gcrs.AlbersEqualArea(),
                       scale='NUMBER OF PERSONS INJURED', limits=(0, 10),
                       legend=True, legend_kwargs={'loc': 'upper left'})

    .. image:: ../figures/pointplot/pointplot-limits.png

    The default scaling function is linear: an observations at the midpoint of two others will be exactly midway
    between them in size. To specify an alternative scaling function, use the ``scale_func`` parameter. This should
    be a factory function of two variables which, when given the maximum and minimum of the dataset,
    returns a scaling function which will be applied to the rest of the data. A demo is available in
    the `example gallery <examples/usa-city-elevations.html>`_.

    .. code-block:: python

        def trivial_scale(minval, maxval):
            def scalar(val):
                return 2
            return scalar

        gplt.pointplot(collisions, projection=gcrs.AlbersEqualArea(),
                       scale='NUMBER OF PERSONS INJURED', scale_func=trivial_scale,
                       legend=True, legend_kwargs={'loc': 'upper left'})

    .. image:: ../figures/pointplot/pointplot-scale-func.png

    ``hue`` and ``scale`` can co-exist.

    .. code-block:: python

        gplt.pointplot(collisions[collisions['BOROUGH'].notnull()],
                       projection=gcrs.AlbersEqualArea(),
                       hue='BOROUGH', categorical=True,
                       scale='NUMBER OF PERSONS INJURED', limits=(0, 10),
                       legend=True, legend_kwargs={'loc': 'upper left'})

    .. image:: ../figures/pointplot/pointplot-hue-scale.png

    In case more than one visual variable is used, control which one appears in the legend using ``legend_var``.

    .. code-block:: python

        gplt.pointplot(collisions[collisions['BOROUGH'].notnull()],
                       projection=gcrs.AlbersEqualArea(),
                       hue='BOROUGH', categorical=True,
                       scale='NUMBER OF PERSONS INJURED', limits=(0, 10),
                       legend=True, legend_kwargs={'loc': 'upper left'},
                       legend_var='scale')

    .. image:: ../figures/pointplot/pointplot-legend-var.png

    """
    # Initialize the figure, if one hasn't been initialized already.
    fig = _init_figure(ax, figsize)

    xs = np.array([p.x for p in df.geometry])
    ys = np.array([p.y for p in df.geometry])

    if projection:
        # Properly set up the projection.
        projection = projection.load(df, {
            'central_longitude': lambda df: np.mean(np.array([p.x for p in df.geometry.centroid])),
            'central_latitude': lambda df: np.mean(np.array([p.y for p in df.geometry.centroid]))
        })

        # Set up the axis.
        if not ax:
            ax = plt.subplot(111, projection=projection)

        # Set extent.
        if extent:
            ax.set_extent(extent)
        else:
            pass  # Default extent.
    else:
        ax = plt.subplot(111)

    # Clean up patches.
    _lay_out_axes(ax, projection)

    # Validate hue input.
    hue = _validate_hue(df, hue)

    # Set legend variable.
    if legend_var is None:
        if hue is not None:
            legend_var = "hue"
        elif scale is not None:
            legend_var = "scale"

    # Generate the coloring information, if needed. Follows one of two schemes, categorical or continuous,
    # based on whether or not ``k`` is specified (``hue`` must be specified for either to work).
    if k is not None:
        # Categorical colormap code path.
        categorical, k, scheme = _validate_buckets(categorical, k, scheme)

        if hue is not None:
            cmap, categories, hue_values = _discrete_colorize(categorical, hue, scheme, k, cmap, vmin, vmax)
            colors = [cmap.to_rgba(v) for v in hue_values]

            # Add a legend, if appropriate.
            if legend and (legend_var != "scale" or scale is None):
                _paint_hue_legend(ax, categories, cmap, legend_labels, legend_kwargs)
        else:
            if 'color' not in kwargs.keys():
                colors = ['steelblue']*len(df)
            else:
                colors = [kwargs['color']]*len(df)
                kwargs.pop('color')
    elif k is None and hue is not None:
        # Continuous colormap code path.
        hue_values = hue
        cmap = _continuous_colormap(hue_values, cmap, vmin, vmax)
        colors = [cmap.to_rgba(v) for v in hue_values]

        # Add a legend, if appropriate.
        if legend and (legend_var != "scale" or scale is None):
            _paint_colorbar_legend(ax, hue_values, cmap, legend_kwargs)

    # Check if the ``scale`` parameter is filled, and use it to fill a ``values`` name.
    if scale:
        if isinstance(scale, str):
            scalar_values = df[scale]
        else:
            scalar_values = scale

        # Compute a scale function.
        dmin, dmax = np.min(scalar_values), np.max(scalar_values)
        if not scale_func:
            dslope = (limits[1] - limits[0]) / (dmax - dmin)
            if np.isinf(dslope):  # Edge case: if dmax, dmin are <=10**-30 or so, will overflow and eval to infinity.
                raise ValueError("The data range provided to the 'scale' variable is too small for the default "
                                 "scaling function. Normalize your data or provide a custom 'scale_func'.")
            dscale = lambda dval: limits[0] + dslope * (dval - dmin)
        else:
            dscale = scale_func(dmin, dmax)

        # Apply the scale function.
        scalar_multiples = np.array([dscale(d) for d in scalar_values])
        sizes = scalar_multiples * 20

        # When a scale is applied, large points will tend to obfuscate small ones. Bringing the smaller
        # points to the front (by plotting them last) is a necessary intermediate step, which is what this bit of
        # code does.
        sorted_indices = np.array(sorted(enumerate(sizes), key=lambda tup: tup[1])[::-1])[:,0].astype(int)
        xs = np.array(xs)[sorted_indices]
        ys = np.array(ys)[sorted_indices]
        sizes = np.array(sizes)[sorted_indices]
        colors = np.array(colors)[sorted_indices]

        # Draw a legend, if appropriate.
        if legend and (legend_var == "scale" or hue is None):
            _paint_carto_legend(ax, scalar_values, legend_values, legend_labels, dscale, legend_kwargs)
    else:
        sizes = kwargs.pop('s') if 's' in kwargs.keys() else 20

    # Draw.
    if projection:
        ax.scatter(xs, ys, transform=ccrs.PlateCarree(), c=colors, s=sizes, **kwargs)
    else:
        ax.scatter(xs, ys, c=colors, s=sizes, **kwargs)

    return ax


def polyplot(df, projection=None,
             extent=None,
             figsize=(8, 6), ax=None,
             edgecolor='black',
             facecolor='None', **kwargs):
    """
    Trivially plots whatever geometries are passed to it. Mostly meant to be used in concert with other,
    more interesting plot types.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        A geographic projection. Must be an instance of an object in the ``geoplot.crs`` module,
        e.g. ``geoplot.crs.PlateCarree()``. This parameter is optional: if left unspecified, a pure unprojected
        ``matplotlib`` object will be returned. For more information refer to the tutorial page on `projections
        <http://localhost:63342/geoplot/docs/_build/html/tutorial/projections.html>`_.
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is unset ``geoplot`` will calculate the plot limits. If an extrema tuple is passed,
        that input will be used instead.
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        A ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot`` instance onto which this plot
        will be graphed. If this parameter is left undefined a new axis will be created and used instead.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.patches.Polygon`` instances (`ref
        <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_).

    Returns
    -------
    AxesSubplot or GeoAxesSubplot instance
        The axis object with the plot on it.

    Examples
    --------

    A trivial example can be created with just a geometry and, optionally, a projection.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea())


    .. image:: ../figures/polyplot/polyplot-initial.png

    However, note that ``polyplot`` is mainly intended to be used in concert with other plot types.

    .. code-block:: python

        ax = gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea())
        gplt.pointplot(collisions[collisions['BOROUGH'].notnull()], projection=gcrs.AlbersEqualArea(),
                       hue='BOROUGH', categorical=True,
                       legend=True, edgecolor='white', linewidth=0.5, legend_kwargs={'loc': 'upper left'},
                       ax=ax)


    .. image:: ../figures/polyplot/polyplot-stacked.png

    Additional keyword arguments are passed to the underlying ``matplotlib.patches.Polygon`` instances (`ref
    <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_).

    .. code-block:: python

        ax = gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea(),
                           linewidth=0, facecolor='lightgray')


    .. image:: ../figures/polyplot/polyplot-kwargs.png
    """
    # Initialize the figure.
    fig = _init_figure(ax, figsize)

    # In this case we can return a `matplotlib` plot directly.

    if projection:
        # Properly set up the projection.
        projection = projection.load(df, {
            'central_longitude': lambda df: np.mean(np.array([p.x for p in df.geometry.centroid])),
            'central_latitude': lambda df: np.mean(np.array([p.y for p in df.geometry.centroid]))
        })

        # Set up the axis.
        if not ax:
            ax = plt.subplot(111, projection=projection)

    else:
        ax = plt.subplot(111)

    # Clean up patches.
    _lay_out_axes(ax, projection)

    # Set extent.
    extrema = _get_envelopes_min_maxes(df.geometry.envelope.exterior)
    _set_extent(ax, projection, extent, extrema)

    # Finally we draw the features.
    if projection:
        for geom in df.geometry:
            features = ShapelyFeature([geom], ccrs.PlateCarree())
            ax.add_feature(features, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
    else:
        for geom in df.geometry:
            try:  # Duck test for MultiPolygon.
                for subgeom in geom:
                    feature = descartes.PolygonPatch(subgeom, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
                    ax.add_patch(feature)
            except (TypeError, AssertionError):  # Shapely Polygon.
                feature = descartes.PolygonPatch(geom, facecolor=facecolor, edgecolor=edgecolor, **kwargs)
                ax.add_patch(feature)

    return ax


def choropleth(df, projection=None,
               hue=None,
               scheme=None, k=5, cmap='Set1', categorical=False, vmin=None, vmax=None,
               legend=False, legend_kwargs=None, legend_labels=None,
               extent=None,
               figsize=(8, 6), ax=None,
               **kwargs):
    """
    A simple aggregation plot based on geometry.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        A geographic projection. Must be an instance of an object in the ``geoplot.crs`` module,
        e.g. ``geoplot.crs.PlateCarree()``. This parameter is optional: if left unspecified, a pure unprojected
        ``matplotlib`` object will be returned. For more information refer to the tutorial page on `projections
        <http://localhost:63342/geoplot/docs/_build/html/tutorial/projections.html>`_.
    hue : None, Series, GeoSeries, iterable, or str
        A data column whose values are to be colorized. Defaults to None, in which case no colormap will be applied.
    categorical : boolean, optional
        Specify this variable to be ``True`` if ``hue`` points to a categorical variable. Defaults to False. Ignored
        if ``hue`` is set to None or not specified.
    scheme : None or {"Quantiles"|"Equal_interval"|"Fisher_Jenks"}, optional
        The scheme which will be used to determine categorical bins for the ``hue`` choropleth. If ``hue`` is
        left unspecified or set to None this variable is ignored.
    k : int or None, optional
        If ``categorical`` is False or left unspecified, this number, set to 5 by default, will determine how
        many bins will exist in the output visualization. If this variable is set to ``None``, a continuous colormap
        will be used.
    cmap : matplotlib color, optional
        The matplotlib colormap to be applied to this dataset (`ref
        <http://matplotlib.org/examples/color/colormaps_reference.html>`_). Defaults to ``viridis``.
    vmin : float, optional
        The value that "bottoms out" the colormap. Data column entries whose value is below this level will be
        colored the same threshold value. Defaults to the minimum value in the dataset.
    vmax : float, optional
        The value that "tops out" the colormap. Data column entries whose value is above this level will be
        colored the same threshold value. Defaults to the maximum value in the dataset.
    legend : boolean, optional
        Whether or not to include a legend in the output plot.
    legend_values : list, optional
        Equal intervals will be used for the "points" in the legend by default. However, particularly if your scale
        is non-linear, oftentimes this isn't what you want. If this variable is provided as well, the values
        included in the input will be used by the legend instead.
    legend_labels : list, optional
        If a legend is specified, this parameter can be used to control what names will be attached to the values.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.pyplot.legend`` instance (`ref
        <http://matplotlib.org/users/legend_guide.html>`_).
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is unset ``geoplot`` will calculate the plot limits. If an extrema tuple is passed,
        that input will be used instead.
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        A ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot`` instance onto which this plot
        will be graphed. If this parameter is left undefined a new axis will be created and used instead.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.patches.Polygon`` instances (`ref
        <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_).

    Returns
    -------
    AxesSubplot or GeoAxesSubplot instance
        The axis object with the plot on it.

    Examples
    --------

    The choropleth is a standard-bearer of the field. To make one yourself, you will need a series of enclosed areas,
    consisting of ``shapely`` ``Polygon`` or ``MultiPolygon`` entities, and a series of data about them that you
    would like to express in color. A basic choropleth requires geometry, a ``hue`` variable, and, optionally,
    a projection.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        gplt.choropleth(polydata, hue='latdep', projection=gcrs.PlateCarree())

    .. image:: ../figures/choropleth/choropleth-initial.png

    Change the colormap with the ``cmap`` parameter.

    .. code-block:: python

        gplt.choropleth(polydata, hue='latdep', projection=gcrs.PlateCarree(), cmap='Blues')

    .. image:: ../figures/choropleth/choropleth-cmap.png

    If your variable of interest is already `categorical
    <http://pandas.pydata.org/pandas-docs/stable/categorical.html>`_, you can specify ``categorical=True`` to
    use the labels in your dataset directly.

    .. code-block:: python

        gplt.choropleth(boroughs, projection=gcrs.AlbersEqualArea(), hue='BoroName', categorical=True)

    .. image:: ../figures/choropleth/choropleth-categorical.png

    To add a legend, specify ``legend``.

    .. code-block:: python

        gplt.choropleth(boroughs, projection=gcrs.AlbersEqualArea(), hue='BoroName',
                        categorical=True, legend=True)

    .. image:: ../figures/choropleth/choropleth-legend.png

    Keyword arguments can be passed to the legend using the ``legend_kwargs`` argument. These arguments, often
    necessary to properly position the legend, will be passed to the underlying `matplotlib Legend instance
    <http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend>`_.

    .. code-block:: python

        gplt.choropleth(boroughs, projection=gcrs.AlbersEqualArea(), hue='BoroName',
                        categorical=True, legend=True, legend_kwargs={'loc': 'upper left'})

    .. image:: ../figures/choropleth/choropleth-legend-kwargs.png

    Additional arguments not in the method signature will be passed as keyword parameters to the underlying
    `matplotlib Polygon patches <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_.

    .. code-block:: python

        gplt.choropleth(boroughs, projection=gcrs.AlbersEqualArea(), hue='BoroName', categorical=True,
                        linewidth=0)

    .. image:: ../figures/choropleth/choropleth-kwargs.png

    Choropleths default to splitting the data into five buckets with approximately equal numbers of observations in
    them. Change the number of buckets by specifying ``k``.

    .. code-block:: python

        gplt.choropleth(census_tracts, hue='mock_data', projection=gcrs.AlbersEqualArea(),
                legend=True, edgecolor='white', linewidth=0.5, legend_kwargs={'loc': 'upper left'},
                k=2)

    .. image:: ../figures/choropleth/choropleth-k.png

    To use a continuous colormap, specify ``k=None``. In this case a colorbar legend will be used.

    .. code-block:: python

        gplt.choropleth(polydata, hue='latdep', cmap='Blues', k=None, legend=True,
                        projection=gcrs.PlateCarree())

    .. image:: ../figures/choropleth/choropleth-k-none.png

    ``legend_labels`` controls the legend labels.

    .. code-block:: python

        gplt.choropleth(census_tracts, hue='mock_data', projection=gcrs.AlbersEqualArea(),
                        edgecolor='white', linewidth=0.5,
                        legend=True, legend_kwargs={'loc': 'upper left'},
                        legend_labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    .. image:: ../figures/choropleth/choropleth-legend-labels.png

    Alternatively, change the scheme used to generate the buckets with the ``scheme`` parameter. ``equal_interval``,
    for example, will generate buckets of equal data distribution size instead.

    .. code-block:: python

        gplt.choropleth(census_tracts, hue='mock_data', projection=gcrs.AlbersEqualArea(),
                legend=True, edgecolor='white', linewidth=0.5, legend_kwargs={'loc': 'upper left'},
                scheme='equal_interval')

    .. image:: ../figures/choropleth/choropleth-scheme.png
    """
    # Initialize the figure.
    fig = _init_figure(ax, figsize)

    if projection:
        projection = projection.load(df, {
            'central_longitude': lambda df: np.mean(np.array([p.x for p in df.geometry.centroid])),
            'central_latitude': lambda df: np.mean(np.array([p.y for p in df.geometry.centroid]))
        })

        # Set up the axis.
        if not ax:
            ax = plt.subplot(111, projection=projection)
    else:
        ax = plt.subplot(111)

    # Clean up patches.
    _lay_out_axes(ax, projection)

    # Set extent.
    extrema = _get_envelopes_min_maxes(df.geometry.envelope.exterior)
    _set_extent(ax, projection, extent, extrema)

    # Format the data to be displayed for input.
    hue = _validate_hue(df, hue)
    if hue is None:
        raise ValueError("No 'hue' specified.")

    # Generate the coloring information, if needed. Follows one of two schemes, categorical or continuous,
    # based on whether or not ``k`` is specified (``hue`` must be specified for either to work).
    if k is not None:
        # Categorical colormap code path.

        # Validate buckets.
        categorical, k, scheme = _validate_buckets(categorical, k, scheme)

        if hue is not None:
            cmap, categories, hue_values = _discrete_colorize(categorical, hue, scheme, k, cmap, vmin, vmax)
            colors = [cmap.to_rgba(v) for v in hue_values]

            # Add a legend, if appropriate.
            if legend:
                _paint_hue_legend(ax, categories, cmap, legend_labels, legend_kwargs)
        else:
            colors = ['steelblue']*len(df)
    elif k is None and hue is not None:
        # Continuous colormap code path.
        hue_values = hue
        cmap = _continuous_colormap(hue_values, cmap, vmin, vmax)
        colors = [cmap.to_rgba(v) for v in hue_values]

        # Add a legend, if appropriate.
        if legend:
            _paint_colorbar_legend(ax, hue_values, cmap, legend_kwargs)

    # Draw the features.
    if projection:
        for color, geom in zip(colors, df.geometry):
            features = ShapelyFeature([geom], ccrs.PlateCarree())
            ax.add_feature(features, facecolor=color, **kwargs)
    else:
        for color, geom in zip(colors, df.geometry):
            try:  # Duck test for MultiPolygon.
                for subgeom in geom:
                    feature = descartes.PolygonPatch(subgeom, facecolor=color, **kwargs)
                    ax.add_patch(feature)
            except (TypeError, AssertionError):  # Shapely Polygon.
                feature = descartes.PolygonPatch(geom, facecolor=color, **kwargs)
                ax.add_patch(feature)

    return ax


def aggplot(df, projection=None,
            hue=None,
            by=None,
            geometry=None,
            nmax=None, nmin=None, nsig=0,
            agg=np.mean,
            cmap='viridis', vmin=None, vmax=None,
            legend=True, legend_kwargs=None,
            extent=None,
            figsize=(8, 6), ax=None,
            **kwargs):
    """
    A minimum-expectations summary plot type which handles mixes of geometry types and missing aggregate geometry data.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        A geographic projection. Must be an instance of an object in the ``geoplot.crs`` module,
        e.g. ``geoplot.crs.PlateCarree()``. This parameter is optional: if left unspecified, a pure unprojected
        ``matplotlib`` object will be returned. For more information refer to the tutorial page on `projections
        <http://localhost:63342/geoplot/docs/_build/html/tutorial/projections.html>`_.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The data column whose entries are being discretely colorized. May be passed in any of a number of flexible
        formats. Defaults to None, in which case no colormap will be applied at all.
    by : iterable or str, optional
        The name of a column within the dataset corresponding with some sort of geometry to aggregate points by.
        Specifying ``by`` kicks ``aggplot`` into convex hull plotting mode.
    geometry : GeoDataFrame or GeoSeries, optional
        A ``geopandas`` object containing geometries. When both ``by`` and ``geometry`` are provided ``aggplot``
        plots in geometry plotting mode, matching points in the ``by`` column with the geometries given by their index
        label in the ``geometry`` column, aggregating those, and plotting the results.
    nmax : int or None, optional
        This variable will only be used if the plot is functioning in quadtree mode. It specifies the
        maximum number of observations that will be contained in each quadrangle; any quadrangle containing more
        than ``nmax`` observations will be forcefully partitioned. ``nmax`` may be left unspecified, in which case
        no maximum splitting rule will be used.
    nmin : int, optional
        This variable will only be used if the plot is functioning in quadtree mode. It specifies the minimum number
        of observations that must be present in each quadtree split for the split to be followed through. For
        example, if we specify a value of 5, partition a quadrangle, and find that it contains a subquadrangle with
        just 4 points inside, this rule will cause the algorithm to return the parent quadrangle instead of its
        children.
    nsig : int, optional
        A floor on the number of observations in an aggregation that gets reported. Aggregations containing fewer than
        ``nsig`` points are not aggregated and are instead returned as white patches, indicative of their status as
        "empty" spaces. Defaults to 0.
    agg : function, optional
        The aggregation ufunc that will be applied to the ``numpy`` array of values for the variable of interest of
        observations inside of each quadrangle. Defaults to ``np.mean``.
    cmap : matplotlib color, optional
        The string representation for a matplotlib colormap to be applied to this dataset. ``hue`` must be non-empty
        for a colormap to be applied at all, so this parameter is ignored otherwise.
    vmin : float, optional
        A strict floor on the value associated with the "bottom" of the colormap spectrum. Data column entries whose
        value is below this level will all be colored by the same threshold value.
    vmax : float, optional
        A strict ceiling on the value associated with the "top" of the colormap spectrum. Data column entries whose
        value is above this level will all be colored by the same threshold value.
    legend : boolean, optional
        Whether or not to include a legend in the output plot. This parameter will be ignored if ``hue`` is set to
        None or left unspecified.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the ``matplotlib`` ``ax.colorbar`` method (`ref
        <http://matplotlib.org/api/colorbar_api.html>`_).
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    gridlines : boolean, optional
        Whether or not to overlay cartopy's computed latitude-longitude gridlines.
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is set to None (default) this method will calculate its own cartographic display region. If
        an extrema tuple is passed---useful if you want to focus on a particular area, for example, or exclude certain
        outliers---that input will be used instead.
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        A ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot`` instance onto which this plot
        will be graphed. If this parameter is left undefined a new axis will be created and used instead.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.patches.Polygon`` instances (`ref
        <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_).

    Returns
    -------
    AxesSubplot or GeoAxesSubplot instance
        The axis object with the plot on it.

    Examples
    --------
    This plot type accepts any geometry, including mixtures of polygons and points, averages the value of a certain
    data parameter at their centroids, and plots the result, using a colormap is the visual variable.

    For the purposes of comparison, this library's ``choropleth`` function takes some sort of data as input,
    polygons as geospatial context, and combines themselves into a colorful map. This is useful if, for example,
    you have data on the amount of crimes committed per neigborhood, and you want to plot that.

    But suppose your original dataset came in terms of individual observations - instead of "n collisions happened
    in this neighborhood", you have "one collision occured at this specific coordinate at this specific date".
    This is obviously more useful data - it can be made to do more things - but in order to generate the same map,
    you will first have to do all of the work of geolocating your points to neighborhoods (not trivial),
    then aggregating them (by, in this case, taking a count).

    ``aggplot`` handles this work for you. It takes input in the form of observations, and outputs as useful as
    possible a visualization of their "regional" statistics. What a "region" corresponds to depends on how much
    geospatial information you can provide.

    If you can't provide *any* geospatial context, ``aggplot`` will output what's known as a quadtree: it will break
    your data down into recursive squares, and use them to aggregate the data. This is a very experimental format,
    is very fiddly to make, and has not yet been optimized for speed; but it provides a useful baseline which
    requires no additional work and can be used to expose interesting geospatial correlations right away. And,
    if you have enough observations, it can be `a pretty good approximation
    <../figures/aggplot/aggplot-initial.png>`_ (collisions in New York City pictured).

    Our first few examples are of just such figures. A simple ``aggplot`` quadtree can be generated with just a
    dataset, a data column of interest, and, optionally, a projection.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        gplt.aggplot(collisions, projection=gcrs.PlateCarree(), hue='LATDEP')

    .. image:: ../figures/aggplot/aggplot-initial.png

    To get the best output, you often need to tweak the ``nmin`` and ``nmax`` parameters, controlling the minimum and
    maximum number of observations per box, respectively, yourself. In this case we'll also choose a different
    `matplotlib colormap <http://matplotlib.org/examples/color/colormaps_reference.html>`_, using the ``cmap``
    parameter.

    .. code-block:: python

        gplt.aggplot(collisions, nmin=20, nmax=500, projection=gcrs.PlateCarree(), hue='LATDEP', cmap='Blues')

    .. image:: ../figures/aggplot/aggplot-quadtree.png

    ``aggplot`` will satisfy the ``nmax`` parameter before trying to satisfy ``nmin``, so you may result in spaces
    without observations, or ones lacking a statistically significant number of observations. This is necessary in
    order to break up "spaces" that the algorithm would otherwise end on. You can control the maximum number of
    observations in the blank spaces using the ``nsig`` parameter.

    .. code-block:: python

        gplt.aggplot(collisions, nmin=20, nmax=500, nsig=5, projection=gcrs.PlateCarree(), hue='LATDEP', cmap='Reds')

    .. image:: ../figures/aggplot/aggplot-quadtree-tuned.png

    You'll have to play around with these parameters to get the clearest picture.

    Usually, however, observations with a geospatial component will be provided with some form of spatial
    categorization. In the case of our collisions example, this comes in the form of a postal zip code. With the
    simple addition of this data column via the ``by`` parameter, our output changes radically, taking advantage of
    the additional context we now have to sort and aggregate our observations by (hopefully) geospatially
    meaningful, if still crude, grouped convex hulls.

    .. code-block:: python


        gplt.aggplot(collisions, projection=gcrs.PlateCarree(), hue='NUMBER OF PERSONS INJURED', cmap='Reds',
                     by='BOROUGH')

    .. image:: ../figures/aggplot/aggplot-hulls.png

    Finally, suppose you actually know exactly the geometries that you would like to aggregate by. Provide these in
    the form of a ``geopandas`` ``GeoSeries``, one whose index matches the values in your ``by`` column (so
    ``BROOKLYN`` matches ``BROOKLYN`` for example), to the ``geometry`` parameter. Your output will now be an
    ordinary choropleth.

    .. code-block:: python

        gplt.aggplot(collisions, projection=gcrs.PlateCarree(), hue='NUMBER OF PERSONS INJURED', cmap='Reds',
                     by='BOROUGH', geometry=boroughs)

    .. image:: ../figures/aggplot/aggplot-by.png

    Observations will be aggregated by average, by default. In our example case, our plot shows that accidents in
    Manhattan tend to result in significantly fewer injuries than accidents occuring in other boroughs.

    Choose which aggregation to use by passing a function to the ``agg`` parameter.

    .. code-block:: python

        gplt.aggplot(collisions, projection=gcrs.PlateCarree(), hue='NUMBER OF PERSONS INJURED', cmap='Reds',
                 geometry=boroughs_2, by='BOROUGH', agg=len)

    .. image:: ../figures/aggplot/aggplot-agg.png

    ``legend`` toggles the legend.

    .. code-block:: python

        gplt.aggplot(collisions, projection=gcrs.PlateCarree(), hue='NUMBER OF PERSONS INJURED', cmap='Reds',
                 geometry=boroughs_2, by='BOROUGH', agg=len, legend=False)

    .. image:: ../figures/aggplot/aggplot-legend.png

    Additional keyword arguments are passed to the underlying ``matplotlib.patches.Polygon`` instances
    (`ref <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_).

    .. code-block:: python

        gplt.aggplot(collisions, projection=gcrs.PlateCarree(), hue='NUMBER OF PERSONS INJURED', cmap='Reds',
                 geometry=boroughs_2, by='BOROUGH', agg=len, linewidth=0)

    .. image:: ../figures/aggplot/aggplot-kwargs.png

    Additional keyword arguments for styling the `colorbar <http://matplotlib.org/api/colorbar_api.html>`_ legend are
    passed using ``legend_kwargs``.

    .. code-block:: python

        gplt.aggplot(collisions, projection=gcrs.PlateCarree(), hue='NUMBER OF PERSONS INJURED', cmap='Reds',
                     geometry=boroughs_2, by='BOROUGH', agg=len, linewidth=0,
                     legend_kwargs={'orientation': 'horizontal'})

    .. image:: ../figures/aggplot/aggplot-legend-kwargs.png
    """
    fig = _init_figure(ax, figsize)

    # Set up projection.
    if projection:
        projection = projection.load(df, {
            'central_longitude': lambda df: np.mean(np.array([p.x for p in df.geometry.centroid])),
            'central_latitude': lambda df: np.mean(np.array([p.y for p in df.geometry.centroid]))
        })

        if not ax:
            ax = plt.subplot(111, projection=projection)
    else:
        ax = plt.subplot(111)

    # Clean up patches.
    _lay_out_axes(ax, projection)

    # Upconvert input to a GeoDataFrame (necessary for quadtree comprehension).
    df = gpd.GeoDataFrame(df, geometry=df.geometry)

    # Validate hue.
    if not isinstance(hue, str):
        hue_col = hash(str(hue))
        df[hue_col] = _validate_hue(df, hue)
    else:
        hue_col = hue

    if geometry is not None and by is None:
        raise NotImplementedError("Aggregation by geometry alone is not currently implemented and unlikely to be "
                                  "implemented in the future - it is likely out-of-scope here due to the algorithmic "
                                  "complexity involved.")
        # The user wants us to classify our data geometries by their location within the passed world geometries
        # ("sectors"), aggregate a statistic based on that, and return a plot. Unfortunately this seems to be too
        # hard for the moment. Two reasons:
        # 1. The Shapely API for doing so is just about as consistent as can be, but still a little bit inconsistent.
        #    In particular, it is not obvious what to do with invalid and self-intersecting geometric components passed
        #    to the algorithm.
        # 2. Point-in-polygon and, worse, polygon-in-polygon algorithms are extremely slow, to the point that almost
        #    any optimizations that the user can make by doing classification "by hand" is worth it.
        # There should perhaps be a separate library or ``geopandas`` function for doing this.

    elif by is not None:

        # Side-convert geometry for ease of use.
        if geometry is not None:
            # Downconvert GeoDataFrame to GeoSeries objects.
            if isinstance(geometry, gpd.GeoDataFrame):
                geometry = geometry.geometry

        sectors = []
        values = []

        # The groupby operation does not take generators as inputs, so we duck test and convert them to lists.
        if not isinstance(by, str):
            try: len(by)
            except TypeError: by = list(by)

        for label, p in df.groupby(by):
            if geometry is not None:
                try:
                    sector = geometry.loc[label]
                except KeyError:
                    raise KeyError("Data contains a '{0}' label which lacks a corresponding value in the provided "
                                   "geometry.".format(label))
            else:
                xs = [c.x for c in p.geometry]
                ys = [c.y for c in p.geometry]
                coords = list(zip(xs, ys))
                sector = shapely.geometry.MultiPoint(coords).convex_hull

            sectors.append(sector)
            values.append(agg(p[hue_col]))

        # Because we have to set the extent ourselves, we have to do some bookkeeping to keep track of the
        # extrema of the hulls we are generating.
        bxmin = bxmax = bymin = bymax = None
        if not extent:
            for sector in sectors:
                if not isinstance(sector.envelope, shapely.geometry.Point):
                    hxmin, hxmax, hymin, hymax = _get_envelopes_min_maxes(pd.Series(sector.envelope.exterior))
                    if not bxmin or hxmin < bxmin:
                        bxmin = hxmin
                    if not bxmax or hxmax > bxmax:
                        bxmax = hxmax
                    if not bymin or hymin < bymin:
                        bymin = hymin
                    if not bymax or hymax > bymax:
                        bymax = hymax

        # By often creates overlapping polygons, to keep smaller polygons from being hidden by possibly overlapping
        # larger ones we have to bring the smaller ones in front in the plotting order. This bit of code does that.
        sorted_indices = np.array(sorted(enumerate(gpd.GeoSeries(sectors).area.values),
                                         key=lambda tup: tup[1])[::-1])[:, 0].astype(int)
        sectors = np.array(sectors)[sorted_indices]
        values = np.array(values)[sorted_indices]

        # Generate a colormap.
        cmap = _continuous_colormap(values, cmap, vmin, vmax)
        colors = [cmap.to_rgba(value) for value in values]

        #  Draw.
        for sector, color in zip(sectors, colors):
            if projection:
                features = ShapelyFeature([sector], ccrs.PlateCarree())
                ax.add_feature(features, facecolor=color, **kwargs)
            else:
                try:  # Duck test for MultiPolygon.
                    for subgeom in sector:
                        feature = descartes.PolygonPatch(subgeom, facecolor=color, **kwargs)
                        ax.add_patch(feature)
                except (TypeError, AssertionError):  # Shapely Polygon.
                    feature = descartes.PolygonPatch(sector, facecolor=color, **kwargs)
                    ax.add_patch(feature)

        # Set extent.
        extrema = (bxmin, bxmax, bymin, bymax)
        _set_extent(ax, projection, extent, extrema)

    else:
        # Set reasonable defaults for the n-params if appropriate.
        nmax = nmax if nmax else len(df)
        nmin = nmin if nmin else np.max([1, np.min([20, int(0.05 * len(df))])])

        # Generate a quadtree.
        quad = QuadTree(df)
        bxmin, bxmax, bymin, bymax = quad.bounds

        # Assert that nmin is not smaller than the largest number of co-located observations (otherwise the algorithm
        # would continue running until the recursion limit).
        max_coloc = np.max([len(l) for l in quad.agg.values()])
        if max_coloc > nmin:
            raise ValueError("nmin is set to {0}, but there is a coordinate containing {1} observations in the "
                             "dataset.".format(nmin, max_coloc))

        # Run the partitions.
        # partitions = quad.partition(nmin, nmax)
        partitions = list(quad.partition(nmin, nmax))

        # Generate colormap.
        values = [agg(p.data[hue_col]) for p in partitions if p.n > nsig]
        cmap = _continuous_colormap(values, cmap, vmin, vmax)

        for p in partitions:
            xmin, xmax, ymin, ymax = p.bounds
            rect = shapely.geometry.Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)])
            color = cmap.to_rgba(agg(p.data[hue_col])) if p.n > nsig else "white"
            if projection:
                feature = ShapelyFeature([rect], ccrs.PlateCarree())
                ax.add_feature(feature, facecolor=color, **kwargs)
            else:
                feature = descartes.PolygonPatch(rect, facecolor=color, **kwargs)
                ax.add_patch(feature)

        # Set extent.
        extrema = (bxmin, bxmax, bymin, bymax)
        _set_extent(ax, projection, extent, extrema)

    # Append a legend, if appropriate.
    if legend:
        _paint_colorbar_legend(ax, values, cmap, legend_kwargs)

    return ax


def cartogram(df, projection=None,
              scale=None, limits=(0.2, 1), scale_func=None, trace=True, trace_kwargs=None,
              hue=None, categorical=False, scheme=None, k=5, cmap='viridis', vmin=None, vmax=None,
              legend=False, legend_values=None, legend_labels=None, legend_kwargs=None, legend_var="scale",
              extent=None,
              figsize=(8, 6), ax=None,
              **kwargs):
    """
    This plot scales the size of polygonal inputs based on the value of a particular data parameter.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        A geographic projection. Must be an instance of an object in the ``geoplot.crs`` module,
        e.g. ``geoplot.crs.PlateCarree()``. This parameter is optional: if left unspecified, a pure unprojected
        ``matplotlib`` object will be returned. For more information refer to the tutorial page on `projections
        <http://localhost:63342/geoplot/docs/_build/html/tutorial/projections.html>`_.
    scale : str or iterable
        A data column whose values will be used to scale the points.
    limits : (min, max) tuple, optional
        The minimum and maximum limits against which the shape will be scaled. Ignored if ``scale`` is not specified.
    scale_func : ufunc, optional
        The function used to scale point sizes. This should be a factory function of two 
        variables, the minimum and maximum values in the dataset, which returns a scaling function which will be 
        applied to the rest of the data. Defaults to a linear scale. A demo is available in the `example 
        gallery <examples/usa-city-elevations.html>`_.
    trace : boolean, optional
        Whether or not to include a trace of the polygon's original outline in the plot result.
    trace_kwargs : dict, optional
        If ``trace`` is set to ``True``, this parameter can be used to adjust the properties of the trace outline. This
        parameter is ignored if trace is ``False``.
    hue : None, Series, GeoSeries, iterable, or str, optional
        A data column whose values are to be colorized. Defaults to None, in which case no colormap will be applied.
    categorical : boolean, optional
        Specify this variable to be ``True`` if ``hue`` points to a categorical variable. Defaults to False. Ignored
        if ``hue`` is set to None or not specified.
    scheme : None or {"Quantiles"|"Equal_interval"|"Fisher_Jenks"}, optional
        The scheme which will be used to determine categorical bins for the ``hue`` choropleth. If ``hue`` is
        left unspecified or set to None this variable is ignored.
    k : int or None, optional
        If ``hue`` is specified and ``categorical`` is False, this number, set to 5 by default, will determine how
        many bins will exist in the output visualization. If ``hue`` is specified and this variable is set to
        ``None``, a continuous colormap will be used. If ``hue`` is left unspecified or set to None this variable is
        ignored.
    cmap : matplotlib color, optional
        The matplotlib colormap to be applied to this dataset (`ref
        <http://matplotlib.org/examples/color/colormaps_reference.html>`_). This parameter is ignored if ``hue`` is not
        specified.
    vmin : float, optional
        The value that "bottoms out" the colormap. Data column entries whose value is below this level will be
        colored the same threshold value. Defaults to the minimum value in the dataset.
    vmax : float, optional
        The value that "tops out" the colormap. Data column entries whose value is above this level will be
        colored the same threshold value. Defaults to the maximum value in the dataset.
    legend : boolean, optional
        Whether or not to include a legend in the output plot.
    legend_values : list, optional
        Equal intervals will be used for the "points" in the legend by default. However, particularly if your scale
        is non-linear, oftentimes this isn't what you want. If this variable is provided as well, the values
        included in the input will be used by the legend instead.
    legend_labels : list, optional
        If a legend is specified, this parameter can be used to control what names will be attached to the values.
    legend_var : "hue" or "scale", optional
        The name of the visual variable for which a legend will be displayed. Does nothing if ``legend`` is False
        or multiple variables aren't used simultaneously.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.pyplot.legend`` instance (`ref
        <http://matplotlib.org/users/legend_guide.html>`_).
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is unset ``geoplot`` will calculate the plot limits. If an extrema tuple is passed,
        that input will be used instead.
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        A ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot`` instance onto which this plot
        will be graphed. If this parameter is left undefined a new axis will be created and used instead.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.patches.Polygon`` instances (`ref
        <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_).

    Returns
    -------
    GeoAxesSubplot instance
        The axis object with the plot on it.

    Examples
    --------
    A cartogram is a plot type which ingests a series of enclosed shapes (``shapely`` ``Polygon`` or ``MultiPolygon``
    entities, in the ``geoplot`` example) and spits out a view of these shapes in which area is distorted according
    to the size of some parameter of interest. These are two types of cartograms, contiguous and non-contiguous
    ones; only the former is implemented in ``geoplot`` at the moment.

    A basic cartogram specifies data, a projection, and a ``scale`` parameter.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        gplt.cartogram(boroughs, scale='Population Density', projection=gcrs.AlbersEqualArea())

    .. image:: ../figures/cartogram/cartogram-initial.png

    The gray outline can be turned off by specifying ``trace``, and a legend can be added by specifying ``legend``.

    .. code-block:: python

        gplt.cartogram(boroughs, scale='Population Density', projection=gcrs.AlbersEqualArea(),
                       trace=False, legend=True)

    .. image:: ../figures/cartogram/cartogram-trace-legend.png

    Keyword arguments can be passed to the legend using the ``legend_kwargs`` argument. These arguments, often
    necessary to properly position the legend, will be passed to the underlying `matplotlib Legend instance
    http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend`_.

    .. code-block:: python

        gplt.cartogram(boroughs, scale='Population Density', projection=gcrs.AlbersEqualArea(),
                       trace=False, legend=True, legend_kwargs={'loc': 'upper left'})

    .. image:: ../figures/cartogram/cartogram-legend-kwargs.png

    Specify an alternative legend display using the ``legend_values`` and ``legend_labels`` parameters.

    .. code-block:: python

        gplt.cartogram(boroughs, scale='Population Density', projection=gcrs.AlbersEqualArea(), legend=True,
               legend_values=[2.32779655e-07, 6.39683197e-07, 1.01364661e-06, 1.17380941e-06, 2.33642596e-06][::-1],
               legend_labels=['Manhattan', 'Brooklyn', 'Queens', 'The Bronx', 'Staten Island'],
               legend_kwargs={'loc': 'upper left'})

    .. image:: ../figures/cartogram/cartogram-legend-labels.png

    Additional arguments to ``cartogram`` will be interpreted as keyword arguments for the scaled polygons,
    using `matplotlib Polygon patch
    <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_ rules.

    .. code-block:: python

        gplt.cartogram(boroughs, scale='Population Density', projection=gcrs.AlbersEqualArea(),
                       edgecolor='darkgreen')

    .. image:: ../figures/cartogram/cartogram-kwargs.png

    Manipulate the outlines use the ``trace_kwargs`` argument, which accepts the same `matplotlib Polygon patch
    <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_ parameters.

    .. code-block:: python

        gplt.cartogram(boroughs, scale='Population Density', projection=gcrs.AlbersEqualArea(),
                       trace_kwargs={'edgecolor': 'lightgreen'})

    .. image:: ../figures/cartogram/cartogram-trace-kwargs.png

    By default, the polygons will be scaled according to the data such that the minimum value is scaled by a factor of
    0.2 while the largest value is left unchanged. Adjust this using the ``limits`` parameter.

    .. code-block:: python

        gplt.cartogram(boroughs, scale='Population Density', projection=gcrs.AlbersEqualArea(),
                       limits=(0.5, 1))

    .. image:: ../figures/cartogram/cartogram-limits.png

    The default scaling function is linear: an observations at the midpoint of two others will be exactly midway
    between them in size. To specify an alternative scaling function, use the ``scale_func`` parameter. This should
    be a factory function of two variables which, when given the maximum and minimum of the dataset,
    returns a scaling function which will be applied to the rest of the data. A demo is available in
    the `example gallery <examples/usa-city-elevations.html>`_.

    .. code-block:: python

        def trivial_scale(minval, maxval):
            def scalar(val):
                return 0.5
            return scalar

        gplt.cartogram(boroughs, scale='Population Density', projection=gcrs.AlbersEqualArea(),
                       limits=(0.5, 1), scale_func=trivial_scale)

    .. image:: ../figures/cartogram/cartogram-scale-func.png

    ``cartogram`` also provides the same ``hue`` visual variable parameters provided by e.g. ``pointplot``. Although
    it's possible for ``hue`` and ``scale`` to refer to different aspects of the data, it's strongly recommended
    to use the same data column for both. For more information on ``hue``-related arguments, refer to e.g. the
    ``pointplot`` `documentation <./pointplot.html>`_.

    .. code-block:: python

        gplt.cartogram(boroughs, scale='Population Density', projection=gcrs.AlbersEqualArea(),
                       hue='Population Density', k=None, cmap='Blues')

    .. image:: ../figures/cartogram/cartogram-hue.png
    """
    # Initialize the figure.
    fig = _init_figure(ax, figsize)

    # Load the projection.
    if projection:
        projection = projection.load(df, {
            'central_longitude': lambda df: np.mean(np.array([p.x for p in df.geometry.centroid])),
            'central_latitude': lambda df: np.mean(np.array([p.y for p in df.geometry.centroid]))
        })

        # Set up the axis.
        if not ax:
            ax = plt.subplot(111, projection=projection)

        # Clean up patches.
    else:
        ax = plt.subplot(111)

    # Clean up patches.
    _lay_out_axes(ax, projection)

    # Set extent.
    extrema = _get_envelopes_min_maxes(df.geometry.envelope.exterior)
    _set_extent(ax, projection, extent, extrema)

    # Check that the ``scale`` parameter is filled, and use it to fill a ``values`` name.
    if not scale:
        raise ValueError("No scale parameter provided.")
    elif isinstance(scale, str):
        values = df[scale]
    else:
        values = scale

    # Compute a scale function.
    dmin, dmax = np.min(values), np.max(values)
    if not scale_func:
        dslope = (limits[1] - limits[0]) / (dmax - dmin)
        dscale = lambda dval: limits[0] + dslope * (dval - dmin)
    else:
        dscale = scale_func(dmin, dmax)

    # Create a legend, if appropriate.
    if legend:
        _paint_carto_legend(ax, values, legend_values, legend_labels, dscale, legend_kwargs)

    # Validate hue input.
    hue = _validate_hue(df, hue)

    # Generate the coloring information, if needed. Follows one of two schemes, categorical or continuous,
    # based on whether or not ``k`` is specified (``hue`` must be specified for either to work).
    if k is not None and hue is not None:
        # Categorical colormap code path.
        categorical, k, scheme = _validate_buckets(categorical, k, scheme)

        if hue is not None:
            cmap, categories, hue_values = _discrete_colorize(categorical, hue, scheme, k, cmap, vmin, vmax)
            colors = [cmap.to_rgba(v) for v in hue_values]

            # Add a legend, if appropriate.
            if legend and (legend_var != "scale" or scale is None):
                _paint_hue_legend(ax, categories, cmap, legend_labels, legend_kwargs)
        else:
            colors = ['None']*len(df)
    elif k is None and hue is not None:
        # Continuous colormap code path.
        hue_values = hue
        cmap = _continuous_colormap(hue_values, cmap, vmin, vmax)
        colors = [cmap.to_rgba(v) for v in hue_values]

        # Add a legend, if appropriate.
        if legend and (legend_var != "scale" or scale is None):
            _paint_colorbar_legend(ax, hue_values, cmap, legend_kwargs)
    elif 'facecolor' in kwargs:
        colors = [kwargs.pop('facecolor')]*len(df)
    else:
        colors = ['None']*len(df)

    # Manipulate trace_kwargs.
    if trace:
        if trace_kwargs is None:
            trace_kwargs = dict()
        if 'edgecolor' not in trace_kwargs.keys():
            trace_kwargs['edgecolor'] = 'lightgray'
        if 'facecolor' not in trace_kwargs.keys():
            trace_kwargs['facecolor'] = 'None'

    # Draw traces first, if appropriate.
    if trace:
        if projection:
            for polygon in df.geometry:
                features = ShapelyFeature([polygon], ccrs.PlateCarree())
                ax.add_feature(features, **trace_kwargs)
        else:
            for polygon in df.geometry:
                try:  # Duck test for MultiPolygon.
                    for subgeom in polygon:
                        feature = descartes.PolygonPatch(subgeom, **trace_kwargs)
                        ax.add_patch(feature)
                except (TypeError, AssertionError):  # Shapely Polygon.
                    feature = descartes.PolygonPatch(polygon, **trace_kwargs)
                    ax.add_patch(feature)

    # Finally, draw the scaled geometries.
    for value, color, polygon in zip(values, colors, df.geometry):
        scale_factor = dscale(value)
        scaled_polygon = shapely.affinity.scale(polygon, xfact=scale_factor, yfact=scale_factor)
        if projection:
            features = ShapelyFeature([scaled_polygon], ccrs.PlateCarree())
            ax.add_feature(features, facecolor=color, **kwargs)
        else:
            try:  # Duck test for MultiPolygon.
                for subgeom in scaled_polygon:
                    feature = descartes.PolygonPatch(subgeom, facecolor=color, **kwargs)
                    ax.add_patch(feature)
            except (TypeError, AssertionError):  # Shapely Polygon.
                feature = descartes.PolygonPatch(scaled_polygon, facecolor=color, **kwargs)
                ax.add_patch(feature)

    return ax


def kdeplot(df, projection=None,
            extent=None,
            figsize=(8, 6), ax=None,
            clip=None,
            **kwargs):
    """
    Geographic kernel density estimate plot.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        A geographic projection. Must be an instance of an object in the ``geoplot.crs`` module,
        e.g. ``geoplot.crs.PlateCarree()``. This parameter is optional: if left unspecified, a pure unprojected
        ``matplotlib`` object will be returned. For more information refer to the tutorial page on `projections
        <http://localhost:63342/geoplot/docs/_build/html/tutorial/projections.html>`_.
    clip : iterable or GeoSeries, optional
        An iterable of geometries that the KDE plot will be clipped to. This is a visual parameter useful for
        "cleaning up" the plot. This feature has not yet actually been implemented!
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is set to None (default) this method will calculate its own cartographic display region. If
        an extrema tuple is passed---useful if you want to focus on a particular area, for example, or exclude certain
        outliers---that input will be used instead.
    clip : None or iterable or GeoSeries, optional
        If this argument is specified, ``kdeplot`` output will be clipped so that the heatmap only appears when it
        is inside the boundaries of the given geometries.
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is unset ``geoplot`` will calculate the plot limits. If an extrema tuple is passed,
        that input will be used instead.
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        A ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot`` instance onto which this plot
        will be graphed. If this parameter is left undefined a new axis will be created and used instead.
    kwargs: dict, optional
        Keyword arguments to be passed to the ``sns.kdeplot`` method doing the plotting (`ref
        <http://seaborn.pydata.org/generated/seaborn.kdeplot.html>`_).

    Returns
    -------
    AxesSubplot or GeoAxesSubplot instance
        The axis object with the plot on it.

    Examples
    --------
    Give it a dataset containing a geometry of ``shapely`` ``Point`` observations, and ``kdeplot`` will return a
    geospatial `kernel density estimate <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ plot
    showing where they are.

    A basic ``kdeplot`` specified data and, optionally, a projection.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        gplt.kdeplot(collisions, projection=gcrs.AlbersEqualArea())

    .. image:: ../figures/kdeplot/kdeplot-initial.png

    However, kdeplots need additional geospatial context to be interpretable. In this case (and for the remainder of
    the examples) we will provide this by overlaying borough geometry.

    .. code-block:: python

        ax = gplt.kdeplot(collisions, projection=gcrs.AlbersEqualArea())
        gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea(), ax=ax)

    .. image:: ../figures/kdeplot/kdeplot-overlay.png

    Most of the rest of the parameters to ``kdeplot`` are parameters inherited from `the seaborn method by the same
    name <http://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot>`_, on which this plot type is
    based. For example, specifying ``shade=True`` provides a filled KDE instead of a contour one:

    .. code-block:: python

        ax = gplt.kdeplot(collisions, projection=gcrs.AlbersEqualArea(),
                          shade=True)
        gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea(), ax=ax)

    .. image:: ../figures/kdeplot/kdeplot-shade.png

    Use ``n_levels`` to specify the number of contour levels.

    .. code-block:: python

        ax = gplt.kdeplot(collisions, projection=gcrs.AlbersEqualArea(),
                          n_levels=30)
        gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea(), ax=ax)

    .. image:: ../figures/kdeplot/kdeplot-n-levels.png

    Or specify ``cmap`` to change the colormap.

    .. code-block:: python

        ax = gplt.kdeplot(collisions, projection=gcrs.AlbersEqualArea(),
             cmap='Purples')
        gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea(), ax=ax)

    .. image:: ../figures/kdeplot/kdeplot-cmap.png

    Oftentimes given the geometry of the location, a "regular" continuous KDEPlot doesn't make sense. We can specify a
    ``clip`` of iterable geometries, which will be used to trim the ``kdeplot`` (note: if you have set ``shade=True`` as
    a parameter you may need to additionally specify ``shade_lowest=False`` to avoid inversion at the edges).

    .. code-block:: python

        gplt.kdeplot(collisions, projection=gcrs.AlbersEqualArea(),
                     shade=True, clip=boroughs)

    .. image:: ../figures/kdeplot/kdeplot-clip.png

    """
    import seaborn as sns  # Immediately fail if no seaborn.
    sns.reset_orig()  # Reset to default style.

    # Initialize the figure.
    fig = _init_figure(ax, figsize)

    # Necessary prior.
    xs = np.array([p.x for p in df.geometry])
    ys = np.array([p.y for p in df.geometry])

    # Load the projection.
    if projection:
        projection = projection.load(df, {
            'central_longitude': lambda df: np.mean(xs),
            'central_latitude': lambda df: np.mean(ys)
        })

        # Set up the axis.
        if not ax:
            ax = plt.subplot(111, projection=projection)
    else:
        ax = plt.subplot(111)

    # Clean up patches.
    _lay_out_axes(ax, projection)

    # Set extent.
    extrema = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
    _set_extent(ax, projection, extent, extrema)

    if projection:
        if clip is None:
            sns.kdeplot(pd.Series([p.x for p in df.geometry]), pd.Series([p.y for p in df.geometry]),
                        transform=ccrs.PlateCarree(), ax=ax, **kwargs)
        else:
            sns.kdeplot(pd.Series([p.x for p in df.geometry]), pd.Series([p.y for p in df.geometry]),
                        transform=ccrs.PlateCarree(), ax=ax, **kwargs)
            clip_geom = _get_clip(ax.get_extent(crs=ccrs.PlateCarree()), clip)
            feature = ShapelyFeature([clip_geom], ccrs.PlateCarree())
            ax.add_feature(feature, facecolor=(1,1,1), linewidth=0, zorder=100)
    else:
        if clip is None:
            sns.kdeplot(pd.Series([p.x for p in df.geometry]), pd.Series([p.y for p in df.geometry]), ax=ax, **kwargs)
        else:
            clip_geom = _get_clip(ax.get_xlim() + ax.get_ylim(), clip)
            polyplot(gpd.GeoSeries(clip_geom),
                     facecolor='white', linewidth=0, zorder=100, extent=ax.get_xlim() + ax.get_ylim(), ax=ax)
            sns.kdeplot(pd.Series([p.x for p in df.geometry]), pd.Series([p.y for p in df.geometry]),
                        ax=ax, **kwargs)
    return ax


def sankey(*args, projection=None,
           start=None, end=None, path=None,
           hue=None, categorical=False, scheme=None, k=5, cmap='viridis', vmin=None, vmax=None,
           legend=False, legend_kwargs=None, legend_labels=None, legend_values=None, legend_var=None,
           extent=None, figsize=(8, 6), ax=None,
           scale=None, limits=(1, 5), scale_func=None,
           **kwargs):
    """
    A geospatial Sankey diagram (flow map).

    Parameters
    ----------
    df : GeoDataFrame, optional.
        The data being plotted. This parameter is optional - it is not needed if ``start`` and ``end`` (and ``hue``,
        if provided) are iterables.
    projection : geoplot.crs object instance, optional
        A geographic projection. Must be an instance of an object in the ``geoplot.crs`` module,
        e.g. ``geoplot.crs.PlateCarree()``. This parameter is optional: if left unspecified, a pure unprojected
        ``matplotlib`` object will be returned. For more information refer to the tutorial page on `projections
        <http://localhost:63342/geoplot/docs/_build/html/tutorial/projections.html>`_.
    start : str or iterable
        Linear starting points: either the name of a column in ``df`` or a self-contained iterable. This parameter is
        required.
    end : str or iterable
        Linear ending points: either the name of a column in ``df`` or a self-contained iterable. This parameter is
        required.
    path : geoplot.crs object instance or iterable, optional
        If this parameter is provided as an iterable, it is assumed to contain the lines that the user wishes to
        draw to connect the points. If this parameter is provided as a projection, that projection will be used for
        determining how the line is plotted. The default is ``ccrs.Geodetic()``, which means
        that the true shortest path will be plotted
        (`great circle distance <https://en.wikipedia.org/wiki/Great-circle_distance>`_); any other choice of
        projection will result in what the shortest path is in that projection instead.
    hue : None, Series, GeoSeries, iterable, or str, optional
        A data column whose values are to be colorized. Defaults to None, in which case no colormap will be applied.
    categorical : boolean, optional
        Specify this variable to be ``True`` if ``hue`` points to a categorical variable. Defaults to False. Ignored
        if ``hue`` is set to None or not specified.
    scheme : None or {"Quantiles"|"Equal_interval"|"Fisher_Jenks"}, optional
        The scheme which will be used to determine categorical bins for the ``hue`` choropleth. If ``hue`` is
        left unspecified or set to None this variable is ignored.
    k : int or None, optional
        If ``hue`` is specified and ``categorical`` is False, this number, set to 5 by default, will determine how
        many bins will exist in the output visualization. If ``hue`` is specified and this variable is set to
        ``None``, a continuous colormap will be used. If ``hue`` is left unspecified or set to None this variable is
        ignored.
    cmap : matplotlib color, optional
        The matplotlib colormap to be applied to this dataset (`ref
        <http://matplotlib.org/examples/color/colormaps_reference.html>`_). This parameter is ignored if ``hue`` is not
        specified.
    vmin : float, optional
        The value that "bottoms out" the colormap. Data column entries whose value is below this level will be
        colored the same threshold value. Defaults to the minimum value in the dataset.
    vmax : float, optional
        The value that "tops out" the colormap. Data column entries whose value is above this level will be
        colored the same threshold value. Defaults to the maximum value in the dataset.
    scale : str or iterable, optional
        A data column whose values will be used to scale the points. Defaults to None, in which case no scaling will be
        applied.
    limits : (min, max) tuple, optional
        The minimum and maximum limits against which the shape will be scaled. Ignored if ``scale`` is not specified.
    scale_func : ufunc, optional
        The function used to scale point sizes. This should be a factory function of two 
        variables, the minimum and maximum values in the dataset, which returns a scaling function which will be 
        applied to the rest of the data. Defaults to a linear scale. A demo is available in the `example 
        gallery <examples/usa-city-elevations.html>`_.
    legend : boolean, optional
        Whether or not to include a legend in the output plot. This parameter will not work if neither ``hue`` nor
        ``scale`` is unspecified.
    legend_values : list, optional
        Equal intervals will be used for the "points" in the legend by default. However, particularly if your scale
        is non-linear, oftentimes this isn't what you want. If this variable is provided as well, the values
        included in the input will be used by the legend instead.
    legend_labels : list, optional
        If a legend is specified, this parameter can be used to control what names will be attached to the values.
    legend_var : "hue" or "scale", optional
        The name of the visual variable for which a legend will be displayed. Does nothing if ``legend`` is False
        or multiple variables aren't used simultaneously.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.pyplot.legend`` instance (`ref
        <http://matplotlib.org/users/legend_guide.html>`_).
    extent : None or (minx, maxx, miny, maxy), optional
        If this parameter is unset ``geoplot`` will calculate the plot limits. If an extrema tuple is passed,
        that input will be used instead.
    figsize : tuple, optional
        An (x, y) tuple passed to ``matplotlib.figure`` which sets the size, in inches, of the resultant plot.
        Defaults to (8, 6), the ``matplotlib`` default global.
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        A ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot`` instance onto which this plot
        will be graphed. If this parameter is left undefined a new axis will be created and used instead.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.lines.Line2D`` instances (`ref
        <http://matplotlib.org/api/lines_api.html#matplotlib.lines.Line2D>`_).

    Returns
    -------
    AxesSubplot or GeoAxesSubplot instance
        The axis object with the plot on it.

    Examples
    --------
    A `Sankey diagram <https://en.wikipedia.org/wiki/Sankey_diagram>`_ is a type of plot useful for visualizing flow
    through a network. Minard's `diagram <https://upload.wikimedia.org/wikipedia/commons/2/29/Minard.png>`_ of
    Napolean's ill-fated invasion of Russia is a classical example. A Sankey diagram is useful when you wish to show
    movement within a network (a `graph <https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)>`_): traffic
    load a road network, for example, or typical airport traffic patterns.

    This plot type is unusual amongst ``geoplot`` types in that it is meant for *two* columns of geography,
    resulting in a slightly different API. A basic ``sankey`` specifies data, ``start`` points, ``end`` points, and,
    optionally, a projection.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        gplt.sankey(mock_data, start='origin', end='destination', projection=gcrs.PlateCarree())

    .. image:: ../figures/sankey/sankey-initial.png

    However, Sankey diagrams need additional geospatial context to be interpretable. In this case (and for the
    remainder of the examples) we will provide this by overlaying world geometry.

    .. code-block:: python

        ax = gplt.sankey(mock_data, start='origin', end='destination', projection=gcrs.PlateCarree())
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-geospatial-context.png

    This function is very ``seaborn``-like in that the usual ``df`` argument is optional. If geometries are provided
    as independent iterables it can be dropped.

    .. code-block:: python

        ax = gplt.sankey(projection=gcrs.PlateCarree(), start=network['from'], end=network['to'])
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-alternative-method-signature.png

    You may be wondering why the lines are curved. By default, the paths followed by the plot are the *actual*
    shortest paths between those two points, in the spherical sense. This is known as `great circle distance
    <https://en.wikipedia.org/wiki/Great-circle_distance>`_. We can see this clearly in an ortographic projection.

    .. code-block:: python

        ax = gplt.sankey(projection=gcrs.Orthographic(), start=network['from'], end=network['to'],
                 extent=(-180, 180, -90, 90))
        ax.set_global()
        ax.coastlines()
        ax.outline_patch.set_visible(True)

    .. image:: ../figures/sankey/sankey-greatest-circle-distance.png

    Plot using a different distance metric, pass it as an argument to the ``path`` parameter. Awkwardly, ``cartopy``
    ``crs`` objects (*not* ``geoplot`` ones) are required.

    .. code-block:: python

        import cartopy.ccrs as ccrs
        ax = gplt.sankey(projection=gcrs.PlateCarree(), start=network['from'], end=network['to'],
                         path=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines()


    .. image:: ../figures/sankey/sankey-path-projection.png

    One of the most powerful ``sankey`` features is that if your data has custom paths, you can use those instead
    with the ``path`` parameter.

    .. code-block:: python

        gplt.sankey(dc, path=dc.geometry, projection=gcrs.AlbersEqualArea(), scale='aadt',
                limits=(0.1, 10))


    .. image:: ../figures/sankey/sankey-path.png

    The ``hue`` parameter colorizes paths based on data.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to', path=PlateCarree(),
                         hue='mock_variable')
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-hue.png

    ``cmap`` changes the colormap.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to',
                         hue='mock_variable', cmap='RdYlBu')
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-cmap.png

    ``legend`` adds a legend.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to',
                         hue='mock_variable', cmap='RdYlBu',
                         legend=True)
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-legend.png

    Pass keyword arguments to the legend with ``legend_kwargs``. This is often necessary for positioning.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to',
                         hue='mock_variable', cmap='RdYlBu',
                         legend=True, legend_kwargs={'bbox_to_anchor': (1.4, 1.0)})
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-legend-kwargs.png

    Specify custom legend labels with ``legend_labels``.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to',
                         hue='mock_variable', cmap='RdYlBu',
                         legend=True, legend_kwargs={'bbox_to_anchor': (1.25, 1.0)},
                         legend_labels=['Very Low', 'Low', 'Average', 'High', 'Very High'])
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-legend-labels.png

    Change the number of bins with ``k``.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to',
                         hue='mock_variable', cmap='RdYlBu',
                         legend=True, legend_kwargs={'bbox_to_anchor': (1.25, 1.0)},
                         k=3)
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-k.png

    Change the binning sceme with ``scheme``.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to',
                         hue='mock_variable', cmap='RdYlBu',
                         legend=True, legend_kwargs={'bbox_to_anchor': (1.25, 1.0)},
                         k=3, scheme='equal_interval')
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-scheme.png

    If your variable of interest is already `categorical
    <http://pandas.pydata.org/pandas-docs/stable/categorical.html>`_, specify ``categorical=True`` to
    use the labels in your dataset directly.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to',
                         hue='above_meridian', cmap='RdYlBu',
                         legend=True, legend_kwargs={'bbox_to_anchor': (1.2, 1.0)},
                         categorical=True)
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-categorical.png

    ``scale`` can be used to enable ``linewidth`` as a visual variable.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to',
                         scale='mock_data',
                         legend=True, legend_kwargs={'bbox_to_anchor': (1.2, 1.0)},
                         color='lightblue')
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-scale.png


    By default, the polygons will be scaled according to the data such that the minimum value is scaled by a factor of
    0.2 while the largest value is left unchanged. Adjust this using the ``limits`` parameter.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to',
                         scale='mock_data', limits=(1, 3),
                         legend=True, legend_kwargs={'bbox_to_anchor': (1.2, 1.0)},
                         color='lightblue')
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-limits.png

    The default scaling function is a linear one. You can change the scaling function to whatever you want by
    specifying a ``scale_func`` input. This should be a factory function of two variables which, when given the
    maximum and minimum of the dataset, returns a scaling function which will be applied to the rest of the data.

    .. code-block:: python

        def trivial_scale(minval, maxval):
            def scalar(val):
                return 2
            return scalar

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                         start='from', end='to',
                         scale='mock_data', scale_func=trivial_scale,
                         legend=True, legend_kwargs={'bbox_to_anchor': (1.1, 1.0)},
                         color='lightblue')
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-scale-func.png

    In case more than one visual variable is used, control which one appears in the legend using ``legend_var``.

    .. code-block:: python

        ax = gplt.sankey(network, projection=gcrs.PlateCarree(),
                 start='from', end='to',
                 scale='mock_data',
                 legend=True, legend_kwargs={'bbox_to_anchor': (1.1, 1.0)},
                 hue='mock_data', legend_var="hue")
        ax.set_global()
        ax.coastlines()

    .. image:: ../figures/sankey/sankey-legend-var.png

    """
    # Validate df.
    if len(args) > 1:
        raise ValueError("Invalid input.")
    elif len(args) == 1:
        df = args[0]
    else:
        df = None  # bind the local name here; initialize in a bit.

    # Validate the rest of the input.
    if ((start is None) or (end is None)) and not hasattr(path, "__iter__"):
        raise ValueError("The 'start' and 'end' parameters must both be specified.")
    if (isinstance(start, str) or isinstance(end, str)) and (df is None):
        raise ValueError("Invalid input.")
    if isinstance(start, str):
        start = df[start]
    elif start is not None:
        start = gpd.GeoSeries(start)
    if isinstance(end, str):
        end = df[end]
    elif end is not None:
        end = gpd.GeoSeries(end)
    if (start is not None) and (end is not None) and hasattr(path, "__iter__"):
        raise ValueError("One of 'start' and 'end' OR 'path' must be specified, but they cannot be specified "
                         "simultaneously.")
    if path is None:  # No path provided.
        path = ccrs.Geodetic()
        path_geoms = None
    elif isinstance(path, str):  # Path is a column in the dataset.
        path_geoms = df[path]
    elif hasattr(path, "__iter__"):  # Path is an iterable.
        path_geoms = gpd.GeoSeries(path)
    else:  # Path is a cartopy.crs object.
        path_geoms = None
    if start is not None and end is not None:
        points = pd.concat([start, end])
    else:
        points = None

    # Set legend variable.
    if legend_var is None:
        if scale is not None:
            legend_var = "scale"
        elif hue is not None:
            legend_var = "hue"

    # After validating the inputs, we are in one of two modes:
    # 1. Projective mode. In this case ``path_geoms`` is None, while ``points`` contains a concatenation of our
    #    points (for use in initializing the plot extents). This case occurs when the user specifies ``start`` and
    #    ``end``, and not ``path``. This is "projective mode" because it means that ``path`` will be a
    #    projection---if one is not provided explicitly, the ``gcrs.Geodetic()`` projection.
    # 2. Path mode. In this case ``path_geoms`` is an iterable of LineString entities to be plotted, while ``points``
    #    is None. This occurs when the user specifies ``path``, and not ``start`` or ``end``. This is path mode
    #    because we will need to plot exactly those paths!

    # At this point we'll initialize the rest of the variables we need. The way that we initialize them is going to
    # depend on which code path we are on. Additionally, we will initialize the `df` variable with a projection
    # dummy, if it has not been initialized already. This `df` will only be used for figuring out the extent,
    # and will be discarded afterwards!
    #
    # Variables we need to generate at this point, and why we need them:
    # 1. (clong, clat) --- To pass this to the projection settings.
    # 2. (xmin. xmax, ymin. ymax) --- To pass this to the extent settings.
    # 3. n --- To pass this to the color array in case no ``color`` is specified.
    if path_geoms is None and points is not None:
        if df is None:
            df = gpd.GeoDataFrame(geometry=points)
        xs = np.array([p.x for p in points])
        ys = np.array([p.y for p in points])
        xmin, xmax, ymin, ymax = np.min(xs), np.max(xs), np.min(ys), np.max(ys)
        clong, clat = np.mean(xs), np.mean(ys)
        n = int(len(points) / 2)
    else:  # path_geoms is an iterable
        path_geoms = gpd.GeoSeries(path_geoms)
        xmin, xmax, ymin, ymax = _get_envelopes_min_maxes(path_geoms.envelope.exterior)
        clong, clat = (xmin + xmax) / 2, (ymin + ymax) / 2
        n = len(path_geoms)

    # Initialize the figure.
    fig = _init_figure(ax, figsize)

    # Load the projection.
    if projection:
        projection = projection.load(df, {
            'central_longitude': lambda df: clong,
            'central_latitude': lambda df: clat
        })

        # Set up the axis.
        if not ax:
            ax = plt.subplot(111, projection=projection)
    else:
        ax = plt.subplot(111, projection=projection)

    # Clean up patches.
    _lay_out_axes(ax, projection)

    # Set extent.
    if projection:
        if extent:
            ax.set_extent(extent)
        else:
            ax.set_extent((xmin, xmax, ymin, ymax))
    else:
        if extent:
            ax.set_xlim((extent[0], extent[1]))
            ax.set_ylim((extent[2], extent[3]))
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    # Generate the coloring information, if needed. Follows one of two schemes, categorical or continuous,
    # based on whether or not ``k`` is specified (``hue`` must be specified for either to work).
    if k is not None:
        # Categorical colormap code path.
        categorical, k, scheme = _validate_buckets(categorical, k, scheme)

        hue = _validate_hue(df, hue)

        if hue is not None:
            cmap, categories, hue_values = _discrete_colorize(categorical, hue, scheme, k, cmap, vmin, vmax)
            colors = [cmap.to_rgba(v) for v in hue_values]

            # Add a legend, if appropriate.
            if legend and (legend_var != "scale" or scale is None):
                _paint_hue_legend(ax, categories, cmap, legend_labels, legend_kwargs)
        else:
            if 'color' not in kwargs.keys():
                colors = ['steelblue'] * n
            else:
                colors = [kwargs['color']] * n
                kwargs.pop('color')
    elif k is None and hue is not None:
        # Continuous colormap code path.
        hue_values = hue
        cmap = _continuous_colormap(hue_values, cmap, vmin, vmax)
        colors = [cmap.to_rgba(v) for v in hue_values]

        # Add a legend, if appropriate.
        if legend and (legend_var != "scale" or scale is None):
            _paint_colorbar_legend(ax, hue_values, cmap, legend_kwargs)

    # Check if the ``scale`` parameter is filled, and use it to fill a ``values`` name.
    if scale:
        if isinstance(scale, str):
            scalar_values = df[scale]
        else:
            scalar_values = scale

        # Compute a scale function.
        dmin, dmax = np.min(scalar_values), np.max(scalar_values)
        if not scale_func:
            dslope = (limits[1] - limits[0]) / (dmax - dmin)
            dscale = lambda dval: limits[0] + dslope * (dval - dmin)
        else:
            dscale = scale_func(dmin, dmax)

        # Apply the scale function.
        scalar_multiples = np.array([dscale(d) for d in scalar_values])
        widths = scalar_multiples * 1

        # Draw a legend, if appropriate.
        if legend and (legend_var == "scale"):
            _paint_carto_legend(ax, scalar_values, legend_values, legend_labels, dscale, legend_kwargs)
    else:
        widths = [1] * n  # pyplot default

    # Allow overwriting visual arguments.
    if 'linestyle' in kwargs.keys():
        linestyle = kwargs['linestyle']; kwargs.pop('linestyle')
    else:
        linestyle = '-'
    if 'color' in kwargs.keys():
        colors = [kwargs['color']]*n; kwargs.pop('color')
    elif 'edgecolor' in kwargs.keys():  # plt.plot uses 'color', mpl.ax.add_feature uses 'edgecolor'. Support both.
        colors = [kwargs['edgecolor']]*n; kwargs.pop('edgecolor')
    if 'linewidth' in kwargs.keys():
        widths = [kwargs['linewidth']]*n; kwargs.pop('linewidth')

    if projection:
        # Duck test plot. The first will work if a valid transformation is passed to ``path`` (e.g. we are in the
        # ``start + ``end`` case), the second will work if ``path`` is an iterable (e.g. we are in the ``path`` case).
        try:
            for origin, destination, color, width in zip(start, end, colors, widths):
                ax.plot([origin.x, destination.x], [origin.y, destination.y], transform=path,
                        linestyle=linestyle, linewidth=width, color=color, **kwargs)
        except TypeError:
            for line, color, width in zip(path_geoms, colors, widths):
                feature = ShapelyFeature([line], ccrs.PlateCarree())
                ax.add_feature(feature, linestyle=linestyle, linewidth=width, edgecolor=color, facecolor='None',
                **kwargs)
    else:
        try:
            for origin, destination, color, width in zip(start, end, colors, widths):
                ax.plot([origin.x, destination.x], [origin.y, destination.y],
                        linestyle=linestyle, linewidth=width, color=color, **kwargs)
        except TypeError:
            for path, color, width in zip(path_geoms, colors, widths):
                # We have to implement different methods for dealing with LineString and MultiLineString objects.
                # This calls for, yep, another duck test.
                try:  # LineString
                    line = mpl.lines.Line2D([coord[0] for coord in path.coords],
                                            [coord[1] for coord in path.coords],
                                            linestyle=linestyle, linewidth=width, color=color, **kwargs)
                    ax.add_line(line)
                except NotImplementedError:  # MultiLineString
                    for line in path:
                        line = mpl.lines.Line2D([coord[0] for coord in line.coords],
                                                [coord[1] for coord in line.coords],
                                                linestyle=linestyle, linewidth=width, color=color, **kwargs)
                        ax.add_line(line)
    return ax

##################
# HELPER METHODS #
##################


def _init_figure(ax, figsize):
    """
    Initializes the ``matplotlib`` ``figure``, one of the first things that every plot must do. No figure is
    initialized (and, consequentially, the ``figsize`` argument is ignored) if a pre-existing ``ax`` is passed to
    the method. This is necessary for ``plt.savefig()`` to work.

    Parameters
    ----------
    ax : None or cartopy GeoAxesSubplot instance
        The current axis, if there is one.
    figsize : (x_dim, y_dim) tuple
        The dimension of the resultant plot.

    Returns
    -------
    None or matplotlib.Figure instance
        Returns either nothing or the underlying ``Figure`` instance, depending on whether or not one is initialized.
    """
    if not ax:
        fig = plt.figure(figsize=figsize)
        return fig

def _get_envelopes_min_maxes(envelopes):
    """
    Returns the extrema of the inputted polygonal envelopes. Used for setting chart extent where appropriate. Note
    tha the ``Quadtree.bounds`` object property serves a similar role.

    Parameters
    ----------
    envelopes : GeoSeries
        The envelopes of the given geometries, as would be returned by e.g. ``data.geometry.envelope.exterior``.

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


def _set_extent(ax, projection, extent, extrema):
    """
    Sets the plot extent.

    Parameters
    ----------
    ax : cartopy.GeoAxesSubplot instance
        The axis whose boundaries are being tweaked.
    projection : None or geoplot.crs instance
        The projection, if one is being used.
    extent : None or (xmin, xmax, ymin, ymax) tuple
        A copy of the ``extent`` top-level parameter, if the user choses to specify their own extent. These values
        will be used if ``extent`` is non-``None``.
    extrema : None or (xmin, xmax, ymin, ymax) tuple
        Plot-calculated extrema. These values, which are calculated in the plot above and passed to this function
        (different plots require different calculations), will be used if a user-provided ``extent`` is not provided.

    Returns
    -------
    None
    """
    if extent and projection:  # Input ``extent`` into set_extent().
        ax.set_extent(extent)
    elif extent and not projection:  # Input ``extent`` into set_ylim, set_xlim.
        xmin, xmax, ymin, ymax = extent
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    if not extent and projection:  # Input ``extrema`` into set_extent.
        xmin, xmax, ymin, ymax = extrema
        ax.set_extent((xmin, xmax, ymin, ymax))
    if not extent and not projection:  # Input ``extrema`` into set_ylim, set_xlim.
        xmin, xmax, ymin, ymax = extrema
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))


def _lay_out_axes(ax, projection):
    """
    ``cartopy`` enables a a transparent background patch and an "outline" patch by default. This short method simply
    hides these extraneous visual features. If the plot is a pure ``matplotlib`` one, it does the same thing by
    removing the axis altogether.

    Parameters
    ----------
    ax : matplotlib.Axes instance
        The ``matplotlib.Axes`` instance being manipulated.
    projection : None or geoplot.crs instance
        The projection, if one is used.

    Returns
    -------
    None
    """
    if projection is not None:
        try:
            ax.background_patch.set_visible(False)
            ax.outline_patch.set_visible(False)
        except AttributeError:  # Testing...
            pass
    else:
        plt.gca().axison = False


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
    required : boolean
        Whether or not this parameter is required for the plot in question.

    Returns
    -------
    hue : iterable
        The ``hue`` parameter input as an iterable.
    """
    if hue is None:
        return None
    elif isinstance(hue, str):
        hue = df[hue]
        return hue
    else:
        return gpd.GeoSeries(hue)


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
    norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def _discrete_colorize(categorical, hue, scheme, k, cmap, vmin, vmax):
    """
    Creates a discrete colormap, either using an already-categorical data variable or by bucketing a non-categorical
    ordinal one. If a scheme is provided we compute a distribution for the given data. If one is not provided we
    assume that the input data is categorical.

    This code makes extensive use of ``geopandas`` choropleth facilities.

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
        value is below this level will all be colored by the same threshold value.
    vmax : float
        A strict cealing on the value associated with the "bottom" of the colormap spectrum. Data column entries whose
        value is above this level will all be colored by the same threshold value.

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
            warnings.warn("Generating a colormap using a categorical column with over 10 individual categories. "
                          "This is not recommended!")
        value_map = {v: i for i, v in enumerate(categories)}
        values = [value_map[d] for d in hue]
    cmap = norm_cmap(values, cmap, mpl.colors.Normalize, mpl.cm, vmin=vmin, vmax=vmax)
    return cmap, categories, values


def _paint_hue_legend(ax, categories, cmap, legend_labels, legend_kwargs):
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
    legend_labels : list, optional
        If a legend is specified, this parameter can be used to control what names will be attached to the values.
    legend_kwargs : dict
        Keyword arguments which will be passed to the matplotlib legend instance on initialization. This parameter
        is provided to allow fine-tuning of legend placement at the top level of a plot method, as legends are very
        finicky.

    Returns
    -------
    None
    """

    # Paint patches.
    patches = []
    for value, cat in enumerate(categories):
        patches.append(mpl.lines.Line2D([0], [0], linestyle="none",
                              marker="o",
                              markersize=10, markerfacecolor=cmap.to_rgba(value)))
    # I can't initialize legend_kwargs as an empty dict() by default because of Python's argument mutability quirks.
    # cf. http://docs.python-guide.org/en/latest/writing/gotchas/. Instead my default argument is None,
    # but that doesn't unpack correctly, necessitating setting and passing an empty dict here. Awkward...
    if not legend_kwargs: legend_kwargs = dict()

    # If we are given labels use those, if we are not just use the categories.
    if legend_labels:
        ax.legend(patches, legend_labels, numpoints=1, fancybox=True, **legend_kwargs)
    else:
        ax.legend(patches, categories, numpoints=1, fancybox=True, **legend_kwargs)


def _paint_carto_legend(ax, values, legend_values, legend_labels, scale_func, legend_kwargs):
    """
    Creates a legend and attaches it to the axis. Meant to be used when a ``legend=True`` parameter is passed.

    Parameters
    ----------
    ax : matplotlib.Axes instance
        The ``matplotlib.Axes`` instance on which a legend is being painted.
    values : list
        A list of values being plotted. May be either a list of int types or a list of unique entities in the
        data column (e.g. as generated via ``numpy.unique(data)``. This parameter is meant to be the same as that
        returned by the ``_discrete_colorize`` method.
    legend_values : list, optional
        If a legend is specified, equal intervals will be used for the "points" in the legend by default. However,
        particularly if your scale is non-linear, oftentimes this isn't what you want. If this variable is provided as
        well, the values included in the input will be used by the legend instead.
    legend_labels : list, optional
        If a legend is specified, this parameter can be used to control what names will be attached to
    scale_func : ufunc
        The scaling function being used.
    legend_kwargs : dict
        Keyword arguments which will be passed to the matplotlib legend instance on initialization. This parameter
        is provided to allow fine-tuning of legend placement at the top level of a plot method, as legends are very
        finicky.

    Returns
    -------
    None.
    """

    # Set up the legend values.
    if legend_values is not None:
        display_values = legend_values
    else:
        display_values = np.linspace(np.max(values), np.min(values), num=5)
    display_labels = legend_labels if (legend_labels is not None) else display_values

    # Paint patche s.
    patches = []
    for value in display_values:
        patches.append(mpl.lines.Line2D([0], [0], linestyle="none",
                       marker="o",
                       markersize=(20*scale_func(value))**(1/2),
                       markerfacecolor='None'))
    if not legend_kwargs: legend_kwargs = dict()
    ax.legend(patches, display_labels, numpoints=1, fancybox=True, **legend_kwargs)


def _paint_colorbar_legend(ax, values, cmap, legend_kwargs):
    """
    Creates a legend and attaches it to the axis. Meant to be used when a ``legend=True`` parameter is passed.

    Parameters
    ----------
    ax : matplotlib.Axes instance
        The ``matplotlib.Axes`` instance on which a legend is being painted.
    values : list
        A list of values being plotted. May be either a list of int types or a list of unique entities in the
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
    if not legend_kwargs: legend_kwargs = dict()
    cmap.set_array(values)
    plt.gcf().colorbar(cmap, ax=ax, **legend_kwargs)


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
    if categorical and (k != 5 or scheme):
        raise ValueError("Invalid input: categorical cannot be specified as True simultaneously with scheme or k "
                         "parameters")
    if k > 10:
        warnings.warn("Generating a choropleth using a categorical column with over 10 individual categories. "
                      "This is not recommended!")
    if not scheme:
        scheme = 'Quantiles'  # This trips it correctly later.
    return categorical, k, scheme


def _get_clip(extent, clip):
    xmin, xmax, ymin, ymax = extent
    # We have to add a little bit of padding to the edges of the box, as otherwise the edges will invert a little,
    # surprisingly.
    rect = shapely.geometry.Polygon([(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)])
    rect = shapely.affinity.scale(rect, xfact=1.05, yfact=1.05)
    for geom in clip:
        rect = rect.symmetric_difference(geom)
    return rect
