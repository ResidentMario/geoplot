"""
This module defines the majority of geoplot functions, including all plot types.
"""

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
from cartopy.mpl.geoaxes import GeoAxesSubplot
import warnings
from geoplot.quad import QuadTree
import shapely.geometry
import pandas as pd
import descartes

try:
    from geopandas.plotting import _mapclassify_choro
except ImportError:
    from geopandas.plotting import __pysal_choro as _mapclassify_choro

__version__ = "0.2.4"


class HueMixin:
    """
    Class container for hue-setter code shared across all plots that support hue.
    """
    def set_hue_values(
        self, color_kwarg='color', default_color='steelblue',
        supports_continuous=True, supports_categorical=True
    ):
        hue = self.kwargs.pop('hue')
        cmap = self.kwargs.pop('cmap')

        if supports_categorical:
            scheme = self.kwargs.pop('scheme')
            k = self.kwargs.pop('k')
        else:
            scheme = None
            k = None

        if color_kwarg in self.kwargs and hue is not None:
            raise ValueError(
                f'Cannot specify both "{color_kwarg}" and "hue" in the same plot.'
            )

        hue = _to_geoseries(self.df, hue)
        if hue is None:  # no colormap
            color = self.kwargs.pop(color_kwarg, default_color)
            colors = [color] * len(self.df)
            categorical = False
            categories = None
            hue_values = None
        elif k is None:  # continuous colormap
            cmap = _continuous_colormap(hue, cmap)
            colors = [cmap.to_rgba(v) for v in hue]
            categorical = False
            categories = None
            hue_values = None
        else:  # categorical colormap
            categorical, scheme = _validate_buckets(self.df, hue, k, scheme)
            categories = None
            if hue is not None:
                cmap, categories, hue_values = _discrete_colorize(
                    categorical, hue, scheme, k, cmap
                )
                colors = [cmap.to_rgba(v) for v in hue_values]

        self.colors = colors
        self.hue = hue
        self.scheme = scheme
        self.k = k
        self.cmap = cmap
        self.categorical = categorical
        self.categories = categories
        self.hue_values = hue_values
        self.color_kwarg = color_kwarg
        self.default_color = default_color


class ScaleMixin:
    """
    Class container for scale-setter code shared across all plots that support scale.
    """
    def set_scale_values(self, size_kwarg=None, default_size=20):
        self.limits = self.kwargs.pop('limits')
        self.scale_func = self.kwargs.pop('scale_func')
        self.scale = self.kwargs.pop('scale')
        self.scale = _to_geoseries(self.df, self.scale)

        if self.scale is not None:
            dmin, dmax = np.min(self.scale), np.max(self.scale)
            if self.scale_func is None:
                dslope = (self.limits[1] - self.limits[0]) / (dmax - dmin)
                # edge case: if dmax, dmin are <=10**-30 or so, will overflow and eval to infinity.
                # TODO: better explain this error
                if np.isinf(dslope): 
                    raise ValueError(
                        "The data range provided to the 'scale' variable is too small for the "
                        "default scaling function. Normalize your data or provide a custom "
                        "'scale_func'."
                    )
                self.dscale = lambda dval: self.limits[0] + dslope * (dval - dmin)
            else:
                self.dscale = self.scale_func(dmin, dmax)

            # Apply the scale function.
            self.sizes = np.array([self.dscale(d) for d in self.scale])

            # When a scale is applied, large observations will tend to obfuscate small ones.
            # Plotting in descending size order, so that smaller values end up on top, helps
            # clean the plot up a bit.
            sorted_indices = np.array(
                sorted(enumerate(self.sizes), key=lambda tup: tup[1])[::-1]
            )[:,0].astype(int)
            self.sizes = np.array(self.sizes)[sorted_indices]
            self.df = self.df.iloc[sorted_indices]

            if hasattr(self, 'colors') and self.colors is not None:
                self.colors = np.array(self.colors)[sorted_indices]
            if hasattr(self, 'partitions') and self.partitions is not None:
                raise NotImplementedError  # quadtree does not support scale param

        else:
            size = self.kwargs.pop(size_kwarg, default_size)
            self.sizes = [size] * len(self.df)


class LegendMixin:
    """
    Class container for legend-builder code shared across all plots that support legend.
    """
    def paint_legend(self, supports_hue=True, supports_scale=False):
        legend = self.kwargs.pop('legend', False)

        legend_kwargs = self.kwargs.pop('legend_kwargs')
        legend_labels = self.kwargs.pop('legend_labels')
        legend_values = self.kwargs.pop('legend_values')

        if supports_hue and supports_scale:
            if self.kwargs['legend_var'] is not None:
                legend_var = self.kwargs['legend_var']
            else:
                legend_var = 'hue' if self.hue is not None else 'scale'
        else:
            legend_var = 'hue'
        self.kwargs.pop('legend_var')

        if legend and legend_var == 'hue':
            if self.k is not None:
                _paint_hue_legend(
                    self.ax, self.categories, self.cmap,
                    legend_labels, self.kwargs, self.color_kwarg, legend_kwargs
                )
            else:  # self.k is None
                _paint_colorbar_legend(self.ax, self.hue, self.cmap, legend_kwargs)

        elif legend and legend_var == 'scale':
            # When hue == None, HueMixIn.set_hue_values sets self.colors, which controls the
            # facecolor of the plot marker, to an n-length array of the chosen (or default) static
            # color. We reuse that color value for the legend.
            #
            # When hue != None, we apply the same colormap applied to the plot markers to the
            # legend markers as well.
            if self.hue is None:
                markerfacecolor = self.colors[0]
            else:
                # TODO: actually apply a colormap
                markerfacecolor = self.colors[0]

            # TODO: set markeredgecolor (requires knowing the main plot edgecolor param)
            # Also markeredgewidth?
            markeredgecolor = 'black'

            if legend_values is None:
                # If the user doesn't specify their own legend_values, apply a reasonable
                # default: a five-point linear array from min to max.
                legend_values = np.linspace(
                    np.max(self.scale), np.min(self.scale), num=5, dtype=self.scale.dtype
                )
            if legend_labels is None:
                # If the user doesn't specify their own legend_labels, apply a reasonable
                # default: the 'g' f-string for the given input value.
                legend_labels = ['{0:g}'.format(value) for value in legend_values]

            # Mutate the matplotlib defaults from frameon=False to frameon=True and from
            # fancybox=False to fancybox=True.
            if legend_kwargs is None:
                legend_kwargs = dict()
            legend_kwargs['frameon'] = legend_kwargs.pop('frameon', False)
            legend_kwargs['fancybox'] = legend_kwargs.pop('fancybox', True)

            patches = []
            for legend_value in legend_values:
                patches.append(
                    mpl.lines.Line2D(
                        [0], [0], linestyle='None',
                        marker="o",
                        markersize=self.dscale(legend_value),
                        markeredgecolor=markeredgecolor,
                        markerfacecolor=markerfacecolor
                    )
                )
            self.ax.legend(patches, legend_labels, numpoints=1, **legend_kwargs)            


class ClipMixin:
    """
    Class container for clip-setter code shared across all plots that support clip.
    """
    def paint_clip(self):
        clip = self.kwargs.pop('clip')
        clip = _to_geoseries(self.df, clip)
        if clip is not None:
            if self.projection is not None:
                clip_geom = _get_clip(self.ax.get_extent(crs=ccrs.PlateCarree()), clip)
                feature = ShapelyFeature([clip_geom], ccrs.PlateCarree())
                self.ax.add_feature(feature, facecolor=(1,1,1), linewidth=0, zorder=2)
            else:
                clip_geom = _get_clip(self.ax.get_xlim() + self.ax.get_ylim(), clip)
                xmin, xmax = self.ax.get_xlim()
                ymin, ymax = self.ax.get_ylim()
                extent = (xmin, ymin, xmax, ymax)
                polyplot(
                    gpd.GeoSeries(clip_geom), facecolor='white', linewidth=0, zorder=2,
                    extent=extent, ax=self.ax
                )


class QuadtreeComputeMixin:
    """
    Class container for computing a quadtree.
    """
    def compute_quadtree(self):
        nmin = self.kwargs.pop('nmin')
        nmax = self.kwargs.pop('nmax')
        hue = self.kwargs.get('hue', None)

        df = gpd.GeoDataFrame(self.df, geometry=self.df.geometry)
        hue = _to_geoseries(df, hue)
        if hue is not None:
            # TODO: what happens in the case of a column name collision?
            df = df.assign(hue_col=hue)

        # set reasonable defaults for the n-params
        nmax = nmax if nmax else len(df)
        nmin = nmin if nmin else np.max([1, np.round(len(df) / 100)]).astype(int)

        # Jitter the points. Otherwise if there are n points sharing the same coordinate, but
        # n_sig < n, the quadtree algorithm will recurse infinitely. Jitter is applied randomly
        # on 10**-5 scale, inducing maximum additive inaccuracy of ~1cm - good enough for the
        # vast majority of geospatial applications. If the meaningful precision of your dataset
        # exceeds 1cm, jitter the points yourself.
        df = df.assign(geometry=_jitter_points(df.geometry))

        # Generate a quadtree.
        quad = QuadTree(df)
        partitions = quad.partition(nmin, nmax)
        self.partitions = list(partitions)


class QuadtreeHueMixin(HueMixin):
    """
    Subclass of HueMixin that provides modified hue-setting code for the quadtree plot.
    """
    def set_hue_values(self, color_kwarg, default_color):
        agg = self.kwargs.pop('agg')
        _df = self.df
        dvals = []

        # construct a new df of aggregated values for the colormap op, reset afterwards
        has_hue = 'hue' in self.kwargs and self.kwargs['hue'] is not None
        for p in self.partitions:
            if len(p.data) == 0:  # empty
                dval = agg(pd.Series([0]))
            elif has_hue:
                dval = agg(p.data.hue_col)
            dvals.append(dval)

        if has_hue:
            self.df = pd.DataFrame({
                self.kwargs['hue']: dvals
            })
        super().set_hue_values(color_kwarg='facecolor', default_color='None')
        self.df = _df

        # apply the special nsig parameter colormap rule
        nsig = self.kwargs.pop('nsig')
        for i, dval in enumerate(dvals):
            if dval < nsig:
                self.colors[i] = 'None'


class Plot:
    def __init__(self, df, **kwargs):
        self.df = df
        self.figsize = kwargs.pop('figsize')
        self.ax = kwargs.pop('ax')
        self.extent = kwargs.pop('extent')
        self.projection = kwargs.pop('projection')

        self.init_axis()
        self.kwargs = kwargs

    def init_axis(self):
        if not self.ax:
            plt.figure(figsize=self.figsize)

        if len(self.df.geometry) == 0:
            extrema = np.array([0, 0, 1, 1])  # default matplotlib plot extent
        else:
            extrema = np.array(self.df.total_bounds)

        extent = _to_geoseries(self.df, self.extent)
        central_longitude = np.mean(extent[[0, 2]]) if extent is not None\
            else np.mean(extrema[[0, 2]])
        central_latitude = np.mean(extent[[1, 3]]) if extent is not None\
            else np.mean(extrema[[1, 3]])

        if self.projection:
            self.projection = self.projection.load(self.df, {
                'central_longitude': central_longitude,
                'central_latitude': central_latitude
            })

            if not self.ax:
                ax = plt.subplot(111, projection=self.projection)

        else:
            if self.ax:
                ax = self.ax
            else:
                ax = plt.gca()

            if isinstance(ax, GeoAxesSubplot):
                self.projection = ax.projection
            else:
                ax.set_aspect('equal')

        if len(self.df.geometry) != 0:
            xmin, ymin, xmax, ymax = extent if extent is not None else extrema

            if xmin < -180 or xmax > 180 or ymin < -90 or ymax > 90:
                raise ValueError(
                    f'geoplot expects input geometries to be in latitude-longitude coordinates, '
                    f'but the values provided include points whose values exceed the maximum '
                    f'or minimum possible longitude or latitude values (-180, -90, 180, 90), '
                    f'indicating that the input data is not in proper latitude-longitude format.'
                )

            xmin, xmax = max(xmin, -180), min(xmax, 180)
            ymin, ymax = max(ymin, -90), min(ymax, 90)

            if self.projection is not None:
                try:
                    ax.set_extent((xmin, xmax, ymin, ymax), crs=ccrs.PlateCarree())
                except ValueError:
                    # This occurs either due to numerical stability errors in cartopy or due
                    # to the extent exceeding the projection parameters. The latter *ought* to
                    # only happen when using the Orthographic projection (maybe others?), which
                    # is on a globe and only shows at most half of the world at a time. So if the
                    # plot extent exceeds the world half in any dimension the extent-setting
                    # operation will fail.
                    #
                    # The default behavior in cartopy is to use a global exntent with
                    # central_latitude and central_longitude as its center. This is the behavior
                    # we will follow in failure cases.
                    if isinstance(self.projection, ccrs.Orthographic):
                        warnings.warn(
                            'Plot extent lies outside of the Orthographic projection\'s '
                            'viewport. Defaulting to global extent.'
                        )
                    else:
                        warnings.warn(
                            'Cound not set plot extent successfully due to numerical instability. '
                            'Try setting extent manually. Defaulting to a global extent.'
                        )
                    pass

                ax.outline_patch.set_visible(False)
            else:
                ax.axison = False
                ax.set_xlim((xmin, xmax))
                ax.set_ylim((ymin, ymax))

        self.ax = ax


def pointplot(
    df, projection=None,
    hue=None, cmap='viridis', k=5, scheme=None,
    scale=None, limits=(1, 5), scale_func=None,
    legend=False, legend_var=None, legend_values=None, legend_labels=None, legend_kwargs=None,
    figsize=(8, 6), extent=None, ax=None, **kwargs
):
    """
    A geospatial scatter plot.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        The projection to use. For reference see
        Working with Projections_.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Hue>`_.
    cmap : matplotlib color, optional
        If ``hue`` is specified, the
        `colormap <http://matplotlib.org/examples/color/colormaps_reference.html>`_ to use.
    k : int or None, optional
        If ``hue`` is specified, the number of color categories to split the data into. For a
        continuous colormap, set this value to ``None``.
    scheme : None or {"quantiles"|"equal_interval"|"fisher_jenks"}, optional
        If ``hue`` is specified, the categorical binning scheme to use.
    scale : str or iterable, optional
        The column in the dataset (or an iterable of some other data) with which to scale output
        points. For a reference on this and the other scale-related parameters that follow, see
        `Customizing Plots#Scale <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Scale>`_.
    limits : (min, max) tuple, optional
        If ``scale`` is set, the minimum and maximum size of the points.
    scale_func : ufunc, optional
        If ``scale`` is set, the function used to determine the size of each point. For reference
        see the
        `Pointplot Scale Functions <https://residentmario.github.io/geoplot/examples/usa-city-elevations.html>`_
        demo.
    legend : boolean, optional
        Whether or not to include a map legend. For a reference on this and the other 
        legend-related parameters that follow, see
        `Customizing Plots#Legend <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_var : "hue" or "scale", optional
        Which variable, ``hue`` or ``scale``, to use in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to 
        `the underlying matplotlib.legend instance <http://matplotlib.org/users/legend_guide.html>`_.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Extent>`_.
    figsize : (x, y) tuple, optional
        Sets the size of the plot figure (in inches).
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        If set, the ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot``
        instance to paint the plot on. Defaults to a new axis.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying `matplotlib.pyplot.scatter instance
        <http://matplotlib.org/api/pyplot_api.html#matplotlib.pyplot.scatter>`_.

    Returns
    -------
    ``AxesSubplot`` or ``GeoAxesSubplot``
        The plot axis.

    Examples
    --------
    The ``pointplot`` is a `geospatial scatter plot 
    <https://en.wikipedia.org/wiki/Scatter_plot>`_ that represents each observation in your dataset
    as a single point on a map. It is simple and easily interpretable plot that is universally
    understood, making it an ideal choice for showing simple pointwise relationships between
    observations.

    ``pointplot`` requires, at a minimum, some points for plotting:

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        import geopandas as gpd
        cities = gpd.read_file(gplt.datasets.get_path('usa_cities'))
        gplt.pointplot(cities)

    .. image:: ../figures/pointplot/pointplot-initial.png


    The ``hue`` parameter accepts applies a colormap to a data column. The ``legend`` parameter
    toggles a legend.

    .. code-block:: python

        gplt.pointplot(cities, projection=gcrs.AlbersEqualArea(), hue='ELEV_IN_FT', legend=True)

    .. image:: ../figures/pointplot/pointplot-legend.png

    Keyword arguments that are not part of the ``geoplot`` API are passed to the underlying
    ``matplotlib.pyplot.scatter`` instance, which can be used to customize the appearance of the
    plot. To pass keyword argument to the ``matplotlib.legend.Legend``, use ``legend_kwargs``
    argument.

    .. code-block:: python

        gplt.pointplot(
            cities, projection=gcrs.AlbersEqualArea(), 
            hue='ELEV_IN_FT',
            legend=True, legend_kwargs={'loc': 'upper left'},
            edgecolor='lightgray', linewidth=0.5
        )

    .. image:: ../figures/pointplot/pointplot-kwargs.png

    Change the colormap using ``cmap``, or the number of color bins using ``k``. To use a
    continuous colormap, set ``k=None``.

    .. code-block:: python

        gplt.pointplot(
            cities, projection=gcrs.AlbersEqualArea(),
            hue='ELEV_IN_FT', k=8, cmap='inferno_r',
            legend=True
        )

    .. image:: ../figures/pointplot/pointplot-k.png

    ``scale`` provides an alternative or additional visual variable. The minimum and maximum size
    of the points can be adjusted to fit your data using the ``limits`` parameter. It is often
    benefitial to combine both ``scale`` and ``hue`` in a single plot. In this case, you can use
    the ``legend_var`` variable to control which visual variable the legend is keyed on.

    .. code-block:: python

        gplt.pointplot(
            cities, projection=gcrs.AlbersEqualArea(), 
            hue='ELEV_IN_FT', scale='ELEV_IN_FT', limits=(0.1, 3), cmap='inferno_r',
            legend=True, legend_var='scale'
        )

    .. image:: ../figures/pointplot/pointplot-scale.png
    """
    class PointPlot(Plot, HueMixin, ScaleMixin, LegendMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.set_hue_values(color_kwarg='color', default_color='steelblue')
            self.set_scale_values(size_kwarg='s', default_size=20)
            self.paint_legend(supports_hue=True, supports_scale=True)

        def draw(self):
            ax = plot.ax
            if len(plot.df.geometry) == 0:
                return ax

            xs = np.array([p.x for p in plot.df.geometry])
            ys = np.array([p.y for p in plot.df.geometry])
            if self.projection:
                ax.scatter(
                    xs, ys, transform=ccrs.PlateCarree(), c=plot.colors, s=plot.sizes,
                    **plot.kwargs
                )
            else:
                ax.scatter(xs, ys, c=plot.colors, s=plot.sizes, **plot.kwargs)
            return ax

    plot = PointPlot(
        df, figsize=figsize, ax=ax, extent=extent, projection=projection,
        hue=hue, scheme=scheme, k=k, cmap=cmap, scale=scale, limits=limits, scale_func=scale_func,
        legend=legend, legend_var=legend_var, legend_values=legend_values,
        legend_labels=legend_labels, legend_kwargs=legend_kwargs, **kwargs
    )
    return plot.draw()


def polyplot(df, projection=None, extent=None, figsize=(8, 6), ax=None, **kwargs):
    """
    A trivial polygonal plot.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        The projection to use. For reference see
        `Working with Projections <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Working%20with%20Projections.ipynb>`_.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Extent>`_.
    figsize : (x, y) tuple, optional
        Sets the size of the plot figure (in inches).
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        If set, the ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot``
        instance to paint the plot on. Defaults to a new axis.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib.patches.Polygon`` objects
        <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_.

    Returns
    -------
    ``AxesSubplot`` or ``GeoAxesSubplot``
        The plot axis.

    Examples
    --------

    The polyplot draws polygons on a map.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        import geopandas as gpd
        boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
        gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea())

    .. image:: ../figures/polyplot/polyplot-initial.png

    ``polyplot`` is intended to act as a basemap for other, more interesting plot types.

    .. code-block:: python

        ax = gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea())
        gplt.pointplot(
            collisions[collisions['BOROUGH'].notnull()], hue='BOROUGH', ax=ax, legend=True
        )

    .. image:: ../figures/polyplot/polyplot-stacked.png
    """
    class PolyPlot(Plot):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)

        def draw(self):
            ax = self.ax
            if len(self.df.geometry) == 0:
                return ax

            edgecolor = kwargs.pop('edgecolor', 'black')
            facecolor = kwargs.pop('facecolor', 'None')
            zorder = kwargs.pop('zorder', -1)

            if self.projection:
                for geom in self.df.geometry:
                    features = ShapelyFeature([geom], ccrs.PlateCarree())
                    ax.add_feature(
                        features, facecolor=facecolor, edgecolor=edgecolor, zorder=zorder,
                        **kwargs
                    )
            else:
                for geom in df.geometry:
                    try:  # Duck test for MultiPolygon.
                        for subgeom in geom:
                            feature = descartes.PolygonPatch(
                                subgeom, facecolor=facecolor, edgecolor=edgecolor, zorder=zorder,
                                **kwargs
                            )
                            ax.add_patch(feature)
                    except (TypeError, AssertionError):  # Shapely Polygon.
                        feature = descartes.PolygonPatch(
                            geom, facecolor=facecolor, edgecolor=edgecolor, zorder=zorder,
                            **kwargs
                        )
                        ax.add_patch(feature)

            return ax

    plot = PolyPlot(df, figsize=figsize, ax=ax, extent=extent, projection=projection, **kwargs)
    return plot.draw()


def choropleth(
    df, projection=None,
    hue=None, cmap='viridis', k=5, scheme=None,
    legend=False, legend_kwargs=None, legend_labels=None, legend_values=None,
    extent=None, figsize=(8, 6), ax=None, **kwargs
):
    """
    A color-mapped area plot.

    Parameters
    ----------
    
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        The projection to use. For reference see
        `Working with Projections <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Working%20with%20Projections.ipynb>`_.
    hue : None, Series, GeoSeries, iterable, or str, required
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Hue>`_.
    cmap : matplotlib color, optional
        The
        `colormap <http://matplotlib.org/examples/color/colormaps_reference.html>`_ to use.
    k : int or None, optional
        The number of color categories to split the data into. For a continuous colormap, set this
        value to ``None``.
    scheme : None or {"quantiles"|"equal_interval"|"fisher_jenks"}, optional
        The categorical binning scheme to use.
    legend : boolean, optional
        Whether or not to include a map legend. For a reference on this and the other 
        legend-related parameters that follow, see
        `Customizing Plots#Legend <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to 
        `the underlying matplotlib.legend instance <http://matplotlib.org/users/legend_guide.html>`_.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Extent>`_.
    figsize : (x, y) tuple, optional
        Sets the size of the plot figure (in inches).
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        If set, the ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot``
        instance to paint the plot on. Defaults to a new axis.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib`` `Polygon patches
        <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_.

    Returns
    -------
    ``AxesSubplot`` or ``GeoAxesSubplot``
        The plot axis.

    Examples
    --------

    A choropleth takes observations that have been aggregated on some meaningful polygonal level
    (e.g. census tract, state, country, or continent) and displays the data to the reader using
    color. It is a well-known plot type, and likeliest the most general-purpose and well-known of
    the specifically spatial plot types.

    A basic choropleth requires polygonal geometries and a ``hue`` variable.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        import geopandas as gpd
        boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
        gplt.choropleth(boroughs, hue='Shape_Area')

    .. image:: ../figures/choropleth/choropleth-initial.png

    Change the colormap using ``cmap``, or the number of color bins using ``k``. To use a
    continuous colormap, set ``k=None``. The ``legend`` parameter toggles the legend.

    .. code-block:: python

        gplt.choropleth(
            contiguous_usa, hue='population', projection=gcrs.AlbersEqualArea(),
            cmap='Greens', k=None, legend=True
        )

    .. image:: ../figures/choropleth/choropleth-cmap.png

    Keyword arguments that are not part of the ``geoplot`` API are passed to the underlying
    ``matplotlib.patches.Polygon`` objects; this can be used to control plot aesthetics. To pass
    keyword argument to the ``matplotlib.legend.Legend``, use the ``legend_kwargs`` argument.

    .. code-block:: python

        gplt.choropleth(
            contiguous_usa, hue='population', projection=gcrs.AlbersEqualArea(),
            edgecolor='white', linewidth=1,
            cmap='Greens', legend=True, legend_kwargs={'loc': 'lower left'}
        )

    .. image:: ../figures/choropleth/choropleth-legend-kwargs.png

    Plots with a categorical colormap can use the ``scheme`` parameter to control how the data gets
    sorted into the ``k`` bins. The default ``quantile`` sorts into an equal number of observations
    per bin, whereas ``equal_interval`` creates bins equal in size. The more complicated
    ``fisher_jenks`` scheme is an intermediate between the two.

    .. code-block:: python

        gplt.choropleth(
            contiguous_usa, hue='population', projection=gcrs.AlbersEqualArea(),
            edgecolor='white', linewidth=1,
            cmap='Greens', legend=True, legend_kwargs={'loc': 'lower left'},
            scheme='fisher_jenks'
        )

    .. image:: ../figures/choropleth/choropleth-scheme.png

    Use ``legend_labels`` and ``legend_values`` to customize the labels and values that appear
    in the legend.

    .. code-block:: python

        gplt.choropleth(
            contiguous_usa, hue='population', projection=gcrs.AlbersEqualArea(),
            edgecolor='white', linewidth=1,
            cmap='Greens', legend=True, legend_kwargs={'loc': 'lower left'},
            scheme='fisher_jenks',
            legend_labels=[
                '<3 million', '3-6.7 million', '6.7-12.8 million',
                '12.8-25 million', '25-37 million'
            ]
        )

    .. image:: ../figures/choropleth/choropleth-labels.png
    """
    if hue is None:
        raise ValueError("No 'hue' specified.")

    class ChoroplethPlot(Plot, HueMixin, LegendMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.set_hue_values(color_kwarg=None, default_color=None)
            self.paint_legend(supports_hue=True, supports_scale=False)

        def draw(self):
            ax = self.ax

            if len(df.geometry) == 0:
                return ax

            if self.projection:
                for color, geom in zip(self.colors, df.geometry):
                    features = ShapelyFeature([geom], ccrs.PlateCarree())
                    ax.add_feature(features, facecolor=color, **self.kwargs)
            else:
                for color, geom in zip(self.colors, df.geometry):
                    try:  # Duck test for MultiPolygon.
                        for subgeom in geom:
                            feature = descartes.PolygonPatch(
                                subgeom, facecolor=color, **self.kwargs
                            )
                            ax.add_patch(feature)
                    except (TypeError, AssertionError):  # Shapely Polygon.
                        feature = descartes.PolygonPatch(
                            geom, facecolor=color, **self.kwargs
                        )
                        ax.add_patch(feature)

            return ax

    plot = ChoroplethPlot(
        df, figsize=figsize, ax=ax, extent=extent, projection=projection,
        hue=hue, scheme=scheme, k=k, cmap=cmap,
        legend=legend, legend_values=legend_values, legend_labels=legend_labels,
        legend_kwargs=legend_kwargs, **kwargs
    )
    return plot.draw()


def quadtree(
    df, projection=None, clip=None,
    hue=None, cmap='viridis', k=5, scheme=None,
    nmax=None, nmin=None, nsig=0, agg=np.mean,
    legend=False, legend_kwargs=None,
    extent=None, figsize=(8, 6), ax=None, **kwargs
):
    """
    A choropleth with point aggregate neighborhoods.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        The projection to use. For reference see
        `Working with Projections <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Working%20with%20Projections.ipynb>`_.
    clip : None or iterable or GeoSeries, optional
        If specified, quadrangles will be clipped to the boundaries of this geometry.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Hue>`_.
    cmap : matplotlib color, optional
        If ``hue`` is specified, the
        `colormap <http://matplotlib.org/examples/color/colormaps_reference.html>`_ to use.
    k : int or None, optional
        The number of color categories to split the data into. For a continuous colormap, set this
        value to ``None``.
    scheme : None or {"quantiles"|"equal_interval"|"fisher_jenks"}, optional
        The categorical binning scheme to use.
    nmax : int or None, optional
        The maximum number of observations in a quadrangle.
    nmin : int, optional
        The minimum number of observations in a quadrangle.
    nsig : int, optional
        The minimum number of observations in a quadrangle. Defaults to 0 (only empty patches are
        removed).
    agg : function, optional
        The aggregation func used for the colormap. Defaults to ``np.mean``.
    legend : boolean, optional
        Whether or not to include a map legend. For a reference on this and the other 
        legend-related parameters that follow, see
        `Customizing Plots#Legend <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to 
        `the underlying matplotlib.legend instance <http://matplotlib.org/users/legend_guide.html>`_.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Extent>`_.
    figsize : (x, y) tuple, optional
        Sets the size of the plot figure (in inches).
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        If set, the ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot``
        instance to paint the plot on. Defaults to a new axis.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib`` `Polygon patches
        <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_.

    Returns
    -------
    ``AxesSubplot`` or ``GeoAxesSubplot``
        The plot axis.

    Examples
    --------
    A quadtree is a tree data structure which splits a space into increasingly small rectangular
    fractals. This plot takes a sequence of point or polygonal geometries as input and builds a
    choropleth out of their centroids, where each region is a fractal quadrangle with at least
    ``nsig`` observations.

    A quadtree demonstrates density quite effectively. It's more flexible than a conventional
    choropleth, and given a sufficiently large number of points `can construct a very detailed
    view of a space <https://i.imgur.com/n2xlycT.png>`_.

    A simple ``quadtree`` specifies a dataset. It's recommended to also set a maximum number of
    observations per bin, ``nmax``. The smaller the ``nmax``, the more detailed the plot (the
    minimum value is 1).

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        collisions = gpd.read_file(gplt.datasets.get_path('nyc_collision_factors'))
        gplt.quadtree(collisions, nmax=1)

    .. image:: ../figures/quadtree/quadtree-initial.png

    Use ``clip`` to clip the result to surrounding geometry. Keyword arguments that are not part
    of the ``geoplot`` API are passed to the underlying ``matplotlib.pyplot.scatter`` instance,
    which can be used to customize the appearance of the plot.

    .. code-block:: python

        gplt.quadtree(
            collisions, nmax=1,
            projection=gcrs.AlbersEqualArea(), clip=boroughs,
            facecolor='lightgray', edgecolor='white'
        )
    
    .. image:: ../figures/quadtree/quadtree-clip.png

    Use ``hue`` to add color as a visual variable to the plot. ``cmap`` controls the colormap
    used. ``legend`` toggles the legend. This type of plot is an effective gauge of distribution:
    the more random the plot output, the more geospatially decorrelated the variable.

    .. code-block:: python

        gplt.quadtree(
            collisions, nmax=1,
            projection=gcrs.AlbersEqualArea(), clip=boroughs,
            hue='NUMBER OF PEDESTRIANS INJURED', cmap='Reds',
            edgecolor='white', legend=True
        )

    .. image:: ../figures/quadtree/quadtree-hue.png

    Change the number of bins by specifying an alternative ``k`` value. To use a continuous
    colormap, explicitly specify ``k=None``.  You can change the binning sceme with ``scheme``.
    The default is ``quantile``, which bins observations into classes of different sizes but the
    same numbers of observations. ``equal_interval`` will creates bins that are the same size, but
    potentially containing different numbers of observations. The more complicated ``fisher_jenks``
    scheme is an intermediate between the two.

    .. code-block:: python

        gplt.quadtree(
            collisions, nmax=1,
            projection=gcrs.AlbersEqualArea(), clip=boroughs,
            hue='NUMBER OF PEDESTRIANS INJURED', cmap='Reds', k=None,
            edgecolor='white', legend=True,
        )

    .. image:: ../figures/quadtree/quadtree-k.png

    Observations will be aggregated by average, by default. Specify an alternative aggregation
    function using the ``agg`` parameter.

    .. code-block:: python

        gplt.quadtree(
            collisions, nmax=1, agg=np.max,
            projection=gcrs.AlbersEqualArea(), clip=boroughs,
            hue='NUMBER OF PEDESTRIANS INJURED', cmap='Reds', k=None
            edgecolor='white', legend=True
        )

    .. image:: ../figures/quadtree/quadtree-agg.png

    A basic quadtree plot can be used as an alternative to ``polyplot`` as the base layer of your
    map.

    .. code-block:: python

        ax = gplt.quadtree(
            collisions, nmax=1,
            projection=gcrs.AlbersEqualArea(), clip=boroughs,
            facecolor='lightgray', edgecolor='white', zorder=0
        )
        gplt.pointplot(collisions, s=1, ax=ax)

    .. image:: ../figures/quadtree/quadtree-basemap.png
    """
    class QuadtreePlot(Plot, QuadtreeComputeMixin, QuadtreeHueMixin, LegendMixin, ClipMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.compute_quadtree()
            self.set_hue_values(color_kwarg='facecolor', default_color='None')
            self.paint_legend(supports_hue=True, supports_scale=False)
            self.paint_clip()

        def draw(self):
            ax = self.ax
            if len(self.df.geometry) == 0:
                return ax

            for p, color in zip(self.partitions, self.colors):
                xmin, xmax, ymin, ymax = p.bounds
                rect = shapely.geometry.Polygon(
                    [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
                )

                if projection:
                    feature = ShapelyFeature([rect], ccrs.PlateCarree())
                    ax.add_feature(
                        feature, facecolor=color, **self.kwargs
                    )
                else:
                    feature = descartes.PolygonPatch(
                        rect, facecolor=color, **self.kwargs
                    )
                    ax.add_patch(feature)

            return ax

    plot = QuadtreePlot(
        df, projection=projection,
        clip=clip,
        hue=hue, scheme=scheme, k=k, cmap=cmap,
        nmax=nmax, nmin=nmin, nsig=nsig,
        agg=agg, legend=legend, legend_kwargs=legend_kwargs, extent=extent, figsize=figsize, ax=ax,
        **kwargs
    )
    return plot.draw()


def cartogram(
    df, projection=None,
    scale=None, limits=(0.2, 1), scale_func=None,
    hue=None, cmap='viridis', k=5, scheme=None,
    legend=False, legend_values=None, legend_labels=None, legend_kwargs=None, legend_var="scale",
    extent=None, figsize=(8, 6), ax=None, **kwargs
):
    """
    A scaling area plot.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        The projection to use. For reference see
        `Working with Projections <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Working%20with%20Projections.ipynb>`_.
    scale : str or iterable, required
        The column in the dataset (or an iterable of some other data) with which to scale output
        points. For a reference on this and the other scale-related parameters that follow, see
        `Customizing Plots#Scale <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Scale>`_.
    limits : (min, max) tuple, optional
        If ``scale`` is set, the minimum and maximum size of the points.
    scale_func : ufunc, optional
        If ``scale`` is set, the function used to determine the size of each point. For reference
        see the
        `Pointplot Scale Functions <https://residentmario.github.io/geoplot/examples/usa-city-elevations.html>`_
        demo.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Hue>`_.
    cmap : matplotlib color, optional
        If ``hue`` is specified, the
        `colormap <http://matplotlib.org/examples/color/colormaps_reference.html>`_ to use.
    k : int or None, optional
        If ``hue`` is specified, the number of color categories to split the data into. For a
        continuous colormap, set this value to ``None``.
    scheme : None or {"quantiles"|"equal_interval"|"fisher_jenks"}, optional
        If ``hue`` is specified, the categorical binning scheme to use.
    legend : boolean, optional
        Whether or not to include a map legend. For a reference on this and the other 
        legend-related parameters that follow, see
        `Customizing Plots#Legend <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_var : "hue" or "scale", optional
        Which variable, ``hue`` or ``scale``, to use in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to 
        `the underlying matplotlib.legend instance <http://matplotlib.org/users/legend_guide.html>`_.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Extent>`_.
    figsize : (x, y) tuple, optional
        Sets the size of the plot figure (in inches).
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        If set, the ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot``
        instance to paint the plot on. Defaults to a new axis.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib`` `Polygon patches
        <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_.

    Returns
    -------
    ``AxesSubplot`` or ``GeoAxesSubplot``
        The plot axis.

    Examples
    --------
    A cartogram distorts (grows or shrinks) polygons on a map according to the magnitude of some
    input data. They are a less common but more visually "poppy" alternative to a choropleth.

    A basic cartogram specifies data, a projection, and a ``scale`` parameter.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        import geopandas as gpd
        contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
        gplt.cartogram(contiguous_usa, scale='population', projection=gcrs.AlbersEqualArea())

    .. image:: ../figures/cartogram/cartogram-initial.png

    Toggle the legend with ``legend``. Keyword arguments can be passed to the legend using the
    ``legend_kwargs`` argument. These arguments will be passed to the underlying
    `matplotlib.legend.Legend
    <http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend>`_ instance.

    .. code-block:: python

        gplt.cartogram(
            contiguous_usa, scale='population', projection=gcrs.AlbersEqualArea(),
            legend=True, legend_kwargs={'loc': 'lower right'}
        )

    .. image:: ../figures/cartogram/cartogram-trace-legend.png

    To add a colormap to the plot, specify ``hue``. Use ``cmap`` to control the colormap used
    and ``k`` to control the number of color bins. In this plot we also add a backing outline
    of the original state shapes, for better geospatial context.

    .. code-block:: python

        ax = gplt.cartogram(
            contiguous_usa, scale='population', projection=gcrs.AlbersEqualArea(),
            legend=True, legend_kwargs={'bbox_to_anchor': (1, 0.9)}, legend_var='hue',
            hue='population', cmap='Greens',
        )
        gplt.polyplot(contiguous_usa, facecolor='lightgray', edgecolor='white', ax=ax)

    .. image:: ../figures/cartogram/cartogram-cmap.png

    Use the ``limits`` parameter to adjust the minimum and maximum scaling factors. You can also
    pass a custom scaling function to ``scale_func`` to apply a different scale to the plot (the
    default scaling function is linear); see the `USA City Elevations demo 
    <https://residentmario.github.io/geoplot/examples/usa-city-elevations.html>`_ for an example.

    .. code-block:: python

        ax = gplt.cartogram(
            contiguous_usa, scale='population', projection=gcrs.AlbersEqualArea(),
            legend=True, legend_kwargs={'bbox_to_anchor': (1, 0.9)}, legend_var='hue',
            hue='population', cmap='Greens',
            limits=(0.5, 1)
        )
        gplt.polyplot(contiguous_usa, facecolor='lightgray', edgecolor='white', ax=ax)

    .. image:: ../figures/cartogram/cartogram-limits.png
    """
    if scale is None:
        raise ValueError("No scale parameter provided.")

    class CartogramPlot(Plot, HueMixin, ScaleMixin, LegendMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.set_scale_values(size_kwarg=None, default_size=None)
            self.set_hue_values(color_kwarg='facecolor', default_color='steelblue')
            self.paint_legend(supports_hue=True, supports_scale=True)

        def draw(self):
            ax = self.ax
            if len(self.df.geometry) == 0:
                return ax

            for value, color, polygon in zip(self.sizes, self.colors, self.df.geometry):
                scale_factor = value
                scaled_polygon = shapely.affinity.scale(
                    polygon, xfact=scale_factor, yfact=scale_factor
                )
                if self.projection is not None:
                    features = ShapelyFeature([scaled_polygon], ccrs.PlateCarree())
                    ax.add_feature(features, facecolor=color, **kwargs)
                else:
                    try:  # Duck test for MultiPolygon.
                        for subgeom in scaled_polygon:
                            feature = descartes.PolygonPatch(
                                subgeom, facecolor=color, **self.kwargs
                            )
                            ax.add_patch(feature)
                    except (TypeError, AssertionError):  # Shapely Polygon.
                        feature = descartes.PolygonPatch(
                            scaled_polygon, facecolor=color, **self.kwargs
                        )
                        ax.add_patch(feature)

            return ax

    plot = CartogramPlot(
        df, projection=projection,
        figsize=figsize, ax=ax, extent=extent,
        scale=scale, limits=limits, scale_func=scale_func,
        hue=hue, scheme=scheme, k=k, cmap=cmap,
        legend=legend, legend_values=legend_values, legend_labels=legend_labels,
        legend_kwargs=legend_kwargs, legend_var=legend_var,
        **kwargs
    )
    return plot.draw()


def kdeplot(
    df, projection=None, extent=None, figsize=(8, 6), ax=None, clip=None, shade_lowest=False,
    **kwargs
):
    """
    A kernel density estimate isochrone plot.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        The projection to use. For reference see
        `Working with Projections <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Working%20with%20Projections.ipynb>`_.
    clip : None or iterable or GeoSeries, optional
        If specified, isochrones will be clipped to the boundaries of this geometry.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Extent>`_.
    figsize : (x, y) tuple, optional
        Sets the size of the plot figure (in inches).
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        If set, the ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot``
        instance to paint the plot on. Defaults to a new axis.
    kwargs: dict, optional
        Keyword arguments to be passed to 
        `the underlying seaborn.kdeplot instance <http://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot>`_.
    Returns
    -------
    ``AxesSubplot`` or ``GeoAxesSubplot``
        The plot axis.

    Examples
    --------
    `Kernel density estimation <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_ is a
    technique that non-parameterically estimates a distribution function for a sample of point
    observations. KDEs are a popular tool for analyzing data distributions; this plot applies this
    technique to the geospatial setting.

    A basic ``kdeplot`` takes pointwise data as input. For interpretability, let's also plot the
    underlying borough geometry.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        import geopandas as gpd
        boroughs = gpd.read_file(gplt.datasets.get_path('nyc_boroughs'))
        collisions = gpd.read_file(gplt.datasets.get_path('nyc_collision_factors'))
        ax = gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea())
        gplt.kdeplot(collisions, ax=ax)

    .. image:: ../figures/kdeplot/kdeplot-initial.png

    ``n_levels`` controls the number of isochrones. ``cmap`` control the colormap.

    .. code-block:: python

        ax = gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea())
        gplt.kdeplot(collisions, n_levels=20, cmap='Reds', ax=ax)

    .. image:: ../figures/kdeplot/kdeplot-shade.png

    ``shade`` toggles shaded isochrones. Use ``clip`` to constrain the plot to the surrounding
    geometry.

    .. code-block:: python

        ax = gplt.polyplot(boroughs, ax=ax, projection=gcrs.AlbersEqualArea())
        gplt.kdeplot(collisions, cmap='Reds', shade=True, clip=boroughs, ax=ax)

    .. image:: ../figures/kdeplot/kdeplot-clip.png

    Additional keyword arguments that are not part of the ``geoplot`` API are passed to
    `the underlying seaborn.kdeplot instance <http://seaborn.pydata.org/generated/seaborn.kdeplot.html#seaborn.kdeplot>`_.
    One of the most useful of these parameters is ``shade_lowest``, which toggles shading on the
    lowest (basal) layer of the kernel density estimate.

    .. code-block:: python

        ax = gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea())
        ax = gplt.kdeplot(collisions, cmap='Reds', shade=True, shade_lowest=True, clip=boroughs)

    .. image:: ../figures/kdeplot/kdeplot-shade-lowest.png
    """
    import seaborn as sns  # Immediately fail if no seaborn.

    class KDEPlot(Plot, HueMixin, LegendMixin, ClipMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.set_hue_values(color_kwarg=None, default_color=None, supports_categorical=False)
            self.paint_legend(supports_hue=True, supports_scale=False)
            self.paint_clip()

        def draw(self):
            shade_lowest = self.kwargs.pop('shade_lowest', False)
            ax = self.ax
            if len(self.df.geometry) == 0:
                return ax

            if self.projection:
                sns.kdeplot(
                    pd.Series([p.x for p in self.df.geometry]),
                    pd.Series([p.y for p in self.df.geometry]),
                    transform=ccrs.PlateCarree(), ax=ax, shade_lowest=shade_lowest, cmap=self.cmap,
                    **self.kwargs
                )
            else:
                sns.kdeplot(
                    pd.Series([p.x for p in self.df.geometry]),
                    pd.Series([p.y for p in self.df.geometry]),
                    ax=ax, shade_lowest=shade_lowest, **self.kwargs
                )
            return ax

    plot = KDEPlot(
        df, projection=projection, extent=extent, figsize=figsize, ax=ax, clip=clip,
        shade_lowest=shade_lowest, **kwargs
    )
    return plot.draw()


def sankey(
    df, projection=None,
    hue=None, cmap='viridis', k=5, scheme=None,
    legend=False, legend_kwargs=None, legend_labels=None, legend_values=None, legend_var=None,
    extent=None, figsize=(8, 6),
    scale=None, scale_func=None, limits=(1, 5),
    ax=None, **kwargs
):
    """
    A spatial Sankey or flow map.

    Parameters
    ----------
    df : GeoDataFrame, optional
        The data being plotted.
    projection : geoplot.crs object instance, optional
        The projection to use. For reference see
        `Working with Projections <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Working%20with%20Projections.ipynb>`_.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Hue>`_.
    cmap : matplotlib color, optional
        If ``hue`` is specified, the
        `colormap <http://matplotlib.org/examples/color/colormaps_reference.html>`_ to use.
    k : int or None, optional
        If ``hue`` is specified, the number of color categories to split the data into. For a
        continuous colormap, set this value to ``None``.
    scheme : None or {"quantiles"|"equal_interval"|"fisher_jenks"}, optional
        If ``hue`` is specified, the categorical binning scheme to use.
    scale : str or iterable, optional
        The column in the dataset (or an iterable of some other data) with which to scale output
        points. For a reference on this and the other scale-related parameters that follow, see
        `Customizing Plots#Scale <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Scale>`_.
    limits : (min, max) tuple, optional
        If ``scale`` is set, the minimum and maximum size of the points.
    scale_func : ufunc, optional
        If ``scale`` is set, the function used to determine the size of each point. For reference
        see the
        `Pointplot Scale Functions <https://residentmario.github.io/geoplot/examples/usa-city-elevations.html>`_
        demo.
    legend : boolean, optional
        Whether or not to include a map legend. For a reference on this and the other 
        legend-related parameters that follow, see
        `Customizing Plots#Legend <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_var : "hue" or "scale", optional
        Which variable, ``hue`` or ``scale``, to use in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to 
        `the underlying matplotlib.legend instance <http://matplotlib.org/users/legend_guide.html>`_.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Extent>`_.
    figsize : (x, y) tuple, optional
        Sets the size of the plot figure (in inches).
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        If set, the ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot``
        instance to paint the plot on. Defaults to a new axis.
    kwargs: dict, optional
        Keyword arguments to be passed to 
        `the underlying matplotlib.lines.Line2D <https://matplotlib.org/api/_as_gen/matplotlib.lines.Line2D.html#matplotlib.lines.Line2D>`_
        instances.

    Returns
    -------
    ``AxesSubplot`` or ``GeoAxesSubplot``
        The plot axis.

    Examples
    --------
    A `Sankey diagram <https://en.wikipedia.org/wiki/Sankey_diagram>`_ visualizes flow through a
    network. It can be used to show the magnitudes of data moving through a system. This plot
    brings the Sankey diagram into the geospatial context; useful for analyzing traffic load a road
    network, for example, or travel volumes between different airports.

    A basic ``sankey`` requires a ``GeoDataFrame`` of ``LineString`` or ``MultiPoint`` geometries.
    For interpretability, these examples also include world geometry.

    .. code-block:: python

        import geoplot as gplt
        import geoplot.crs as gcrs
        import geopandas as gpd
        la_flights = gpd.read_file(gplt.datasets.get_path('la_flights'))
        world = gpd.read_file(gplt.datasets.get_path('world'))

        ax = gplt.sankey(la_flights, projection=gcrs.Mollweide())
        gplt.polyplot(world, ax=ax, facecolor='lightgray', edgecolor='white')
        ax.set_global(); ax.outline_patch.set_visible(True)

    .. image:: ../figures/sankey/sankey-geospatial-context.png

    ``hue`` adds color gradation to the map. Use ``cmap`` to control the colormap used and k``
    to control the number of color bins. ``legend`` toggles a legend.

    .. code-block:: python

        ax = gplt.sankey(
            la_flights, projection=gcrs.Mollweide(),
            scale='Passengers', hue='Passengers', cmap='Greens', legend=True
        )
        gplt.polyplot(
            world, ax=ax, facecolor='lightgray', edgecolor='white'
        )
        ax.set_global(); ax.outline_patch.set_visible(True)

    .. image:: ../figures/sankey/sankey-cmap.png

    ``scale`` adds volumetric scaling to the plot. ``limits`` can be used to control the minimum
    and maximum line width.

    .. code-block:: python

        ax = gplt.sankey(
            la_flights, projection=gcrs.Mollweide(),
            scale='Passengers', limits=(1, 10),
            hue='Passengers', cmap='Greens', legend=True
        )
        gplt.polyplot(
            world, ax=ax, facecolor='lightgray', edgecolor='white'
        )
        ax.set_global(); ax.outline_patch.set_visible(True)

    .. image:: ../figures/sankey/sankey-scale.png

    Keyword arguments can be passed to the legend using the ``legend_kwargs`` argument. These
    arguments will be passed to the underlying ``matplotlib`` `Legend
    <http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend>`_. The ``loc`` and
    ``bbox_to_anchor`` parameters are particularly useful for positioning the legend.

    .. code-block:: python

        ax = gplt.sankey(
            la_flights, projection=gcrs.Mollweide(),
            scale='Passengers', limits=(1, 10),
            hue='Passengers', cmap='Greens',
            legend=True, legend_kwargs={'loc': 'lower left'}
        )
        gplt.polyplot(
            world, ax=ax, facecolor='lightgray', edgecolor='white'
        )
        ax.set_global(); ax.outline_patch.set_visible(True)

    .. image:: ../figures/sankey/sankey-legend-kwargs.png

    Sankey plots are an attractive option for network data, such as this dataset of DC roads by
    traffic volume.

    .. code-block:: python 
        
        gplt.sankey(
            dc, scale='aadt', edgecolor='black', limits=(0.1, 10),
            projection=gcrs.AlbersEqualArea()
        )

    .. image:: ../figures/sankey/sankey-dc.png
    """
    class SankeyPlot(Plot, HueMixin, ScaleMixin, LegendMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.set_hue_values(color_kwarg='edgecolor', default_color='steelblue')
            self.set_scale_values(size_kwarg='linewidth', default_size=1)
            self.paint_legend(supports_hue=True, supports_scale=True)

        def draw(self):
            ax = self.ax

            if len(df.geometry) == 0:
                return ax

            def parse_geom(geom):
                if isinstance(geom, shapely.geometry.LineString):
                    return geom
                elif isinstance(geom, shapely.geometry.MultiLineString):
                    return geom
                elif isinstance(geom, shapely.geometry.MultiPoint):
                    return shapely.geometry.LineString(geom)
                else:
                    raise ValueError(
                        f'df.geometry must contain LineString, MultiLineString, or MultiPoint '
                        f'geometries, but an instance of {type(geom)} was found instead.'
                    )
            path_geoms = self.df.geometry.map(parse_geom)

            if 'linestyle' in self.kwargs and 'linestyle' is not None:
                linestyle = kwargs.pop('linestyle')
            else:
                linestyle = '-'

            if self.projection:
                for line, color, width in zip(path_geoms, self.colors, self.sizes):
                    feature = ShapelyFeature([line], ccrs.PlateCarree())
                    ax.add_feature(
                        feature, linestyle=linestyle, linewidth=width, edgecolor=color,
                        facecolor='None', **self.kwargs
                    )
            else:
                for path, color, width in zip(path_geoms, self.colors, self.sizes):
                    # We have to implement different methods for dealing with LineString and
                    # MultiLineString objects.
                    try:  # LineString
                        line = mpl.lines.Line2D(
                            [coord[0] for coord in path.coords],
                            [coord[1] for coord in path.coords],
                            linestyle=linestyle, linewidth=width, color=color,
                            **self.kwargs
                        )
                        ax.add_line(line)
                    except NotImplementedError:  # MultiLineString
                        for line in path:
                            line = mpl.lines.Line2D(
                                [coord[0] for coord in line.coords],
                                [coord[1] for coord in line.coords],
                                linestyle=linestyle, linewidth=width, color=color,
                                **self.kwargs
                            )
                            ax.add_line(line)
            return ax

    plot = SankeyPlot(
        df, figsize=figsize, ax=ax, extent=extent, projection=projection,
        scale=scale, limits=limits, scale_func=scale_func,
        hue=hue, scheme=scheme, k=k, cmap=cmap,
        legend=legend, legend_values=legend_values, legend_labels=legend_labels,
        legend_kwargs=legend_kwargs, legend_var=legend_var,
        **kwargs
    )
    return plot.draw()


def voronoi(
    df, projection=None, clip=None,
    hue=None, cmap='viridis', k=5, scheme=None,
    legend=False, legend_kwargs=None, legend_labels=None, legend_values=True,
    extent=None, edgecolor='black', figsize=(8, 6), ax=None, **kwargs
):
    """
    A geospatial Voronoi diagram.

    Parameters
    ----------
    df : GeoDataFrame
        The data being plotted.
    projection : geoplot.crs object instance, optional
        The projection to use. For reference see
        `Working with Projections <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Working%20with%20Projections.ipynb>`_.
    clip : None or iterable or GeoSeries, optional
        If specified, the output will be clipped to the boundaries of this geometry.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Hue>`_.
    cmap : matplotlib color, optional
        If ``hue`` is specified, the
        `colormap <http://matplotlib.org/examples/color/colormaps_reference.html>`_ to use.
    k : int or None, optional
        If ``hue`` is specified, the number of color categories to split the data into. For a
        continuous colormap, set this value to ``None``.
    scheme : None or {"quantiles"|"equal_interval"|"fisher_jenks"}, optional
        If ``hue`` is specified, the categorical binning scheme to use.
    scale : str or iterable, optional
        The column in the dataset (or an iterable of some other data) with which to scale output
        points. For a reference on this and the other scale-related parameters that follow, see
        `Customizing Plots#Scale <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Scale>`_.
    limits : (min, max) tuple, optional
        If ``scale`` is set, the minimum and maximum size of the points.
    scale_func : ufunc, optional
        If ``scale`` is set, the function used to determine the size of each point. For reference
        see the
        `Pointplot Scale Functions <https://residentmario.github.io/geoplot/examples/usa-city-elevations.html>`_
        demo.
    legend : boolean, optional
        Whether or not to include a map legend. For a reference on this and the other 
        legend-related parameters that follow, see
        `Customizing Plots#Legend <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_var : "hue" or "scale", optional
        Which variable, ``hue`` or ``scale``, to use in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to 
        `the underlying matplotlib.legend instance <http://matplotlib.org/users/legend_guide.html>`_.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent <https://nbviewer.jupyter.org/github/ResidentMario/geoplot/blob/master/notebooks/tutorials/Customizing%20Plots.ipynb#Extent>`_.
    figsize : (x, y) tuple, optional
        Sets the size of the plot figure (in inches).
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        If set, the ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot``
        instance to paint the plot on. Defaults to a new axis.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying ``matplotlib`` `Line2D objects
        <http://matplotlib.org/api/lines_api.html#matplotlib.lines.Line2D>`_.

    Returns
    -------
    ``AxesSubplot`` or ``GeoAxesSubplot``
        The plot axis.

    Examples
    --------

    The `Voronoi region <https://en.wikipedia.org/wiki/Voronoi_diagram>`_ of an point is the set
    of points which is closer to that point than to any other observation in a dataset. A Voronoi
    diagram is a space-filling diagram that constructs all of the Voronoi regions of a dataset and
    plots them. 
    
    Voronoi plots are efficient for judging point density and, combined with colormap, can be used
    to infer regional trends in a set of data.

    A basic ``voronoi`` specifies some point data. We overlay geometry to aid interpretability.

    .. code-block:: python

        ax = gplt.voronoi(injurious_collisions.head(1000))
        gplt.polyplot(boroughs, ax=ax)

    .. image:: ../figures/voronoi/voronoi-simple.png

    ``hue`` parameterizes the color, and ``cmap`` controls the colormap.

    .. code-block:: python

        ax = gplt.voronoi(
            injurious_collisions.head(1000), hue='NUMBER OF PERSONS INJURED', cmap='Reds'
        )
        gplt.polyplot(boroughs, ax=ax)

    .. image:: ../figures/voronoi/voronoi-cmap.png

    Add a ``clip`` of iterable geometries to trim the ``voronoi`` against local geography.

    .. code-block:: python

        ax = gplt.voronoi(
            injurious_collisions.head(1000), hue='NUMBER OF PERSONS INJURED', cmap='Reds',
            clip=boroughs
        )
        gplt.polyplot(boroughs, ax=ax)

    .. image:: ../figures/voronoi/voronoi-clip.png

    ``legend`` adds a a ``matplotlib`` `Legend
    <http://matplotlib.org/api/legend_api.html#matplotlib.legend.Legend>`_. This can be tuned
    even further using the ``legend_kwargs`` argument. Other keyword parameters are passed to the
    underlying ``matplotlib``
    `Polygon patches <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_.

    .. code-block:: python

        ax = gplt.voronoi(
            injurious_collisions.head(1000), hue='NUMBER OF PERSONS INJURED',
            cmap='Reds', clip=boroughs,
            legend=True, legend_kwargs={'loc': 'upper left'},
            linewidth=0.5, edgecolor='white'
        )
        gplt.polyplot(boroughs, ax=ax)

    .. image:: ../figures/voronoi/voronoi-kwargs.png

    Change the number of bins by specifying an alternative ``k`` value. To use a continuous
    colormap, explicitly specify ``k=None``.  You can change the binning sceme with ``scheme``.
    The default is ``quantile``, which bins observations into classes of different sizes but the
    same numbers of observations. ``equal_interval`` will creates bins that are the same size, but
    potentially containing different numbers of observations. The more complicated ``fisher_jenks``
    scheme is an intermediate between the two.

    .. code-block:: python

        ax = gplt.voronoi(
            injurious_collisions.head(1000),
            hue='NUMBER OF PERSONS INJURED', cmap='Reds', k=5, scheme='fisher_jenks',
            clip=boroughs,
            legend=True, legend_kwargs={'loc': 'upper left'},
            linewidth=0.5, edgecolor='white'
        )
        gplt.polyplot(boroughs, ax=ax)

    .. image:: ../figures/voronoi/voronoi-scheme.png

    ``geoplot`` will automatically do the right thing if your variable of interest is already
    `categorical <http://pandas.pydata.org/pandas-docs/stable/categorical.html>`_:

    .. code-block:: python

        ax = gplt.voronoi(
            injurious_collisions.head(1000), hue='NUMBER OF PERSONS INJURED',
            cmap='Reds',
            edgecolor='white', clip=boroughs,
            linewidth=0.5
        )
        gplt.polyplot(boroughs, linewidth=1, ax=ax)

    .. image:: ../figures/voronoi/voronoi-multiparty.png
    """
    class VoronoiPlot(Plot, HueMixin, LegendMixin, ClipMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.set_hue_values(color_kwarg='facecolor', default_color='None')
            self.paint_legend(supports_hue=True, supports_scale=False)
            self.paint_clip()

        def draw(self):
            ax = self.ax
            if len(df.geometry) == 0:
                return ax

            geoms = _build_voronoi_polygons(self.df)
            if self.projection:
                for color, geom in zip(self.colors, geoms):
                    features = ShapelyFeature([geom], ccrs.PlateCarree())
                    ax.add_feature(features, facecolor=color, edgecolor=edgecolor, **self.kwargs)
            else:
                for color, geom in zip(plot.colors, geoms):
                    feature = descartes.PolygonPatch(
                        geom, facecolor=color, edgecolor=edgecolor, **self.kwargs
                    )
                    ax.add_patch(feature)

            return ax

    plot = VoronoiPlot(
        df, figsize=figsize, ax=ax, extent=extent, projection=projection,
        hue=hue, scheme=scheme, k=k, cmap=cmap,
        legend=legend, legend_values=legend_values, legend_labels=legend_labels,
        legend_kwargs=legend_kwargs,
        clip=clip,
        **kwargs
    )
    return plot.draw()


##################
# HELPER METHODS #
##################


def _to_geoseries(df, var):
    """
    Some top-level parameters present in most plot types accept a variety of iterables as input
    types. This method condenses this variety into a single preferred format - a GeoSeries.
    """
    if var is None:
        return None
    elif isinstance(var, str):
        var = df[var]
        return var
    elif isinstance(var, gpd.GeoDataFrame):
        return var.geometry
    else:
        return gpd.GeoSeries(var)


def _continuous_colormap(hue, cmap):
    """
    Creates a continuous colormap.
    """
    mn = min(hue)
    mx = max(hue)
    norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
    return mpl.cm.ScalarMappable(norm=norm, cmap=cmap)


def _discrete_colorize(categorical, hue, scheme, k, cmap):
    """
    Creates a discrete colormap, either using an already-categorical data variable or by bucketing
    a non-categorical ordinal one. If a scheme is provided we compute a distribution for the given
    data. If one is not provided we assume that the input data is categorical.
    """
    if not categorical:
        binning = _mapclassify_choro(hue, scheme, k=k)
        values = binning.yb
        binedges = [binning.yb.min()] + binning.bins.tolist()

        categories = [
            '{0:g} - {1:g}'.format(binedges[i], binedges[i + 1])
            for i in range(len(binedges) - 1)
        ]
    else:
        categories = np.unique(hue)
        value_map = {v: i for i, v in enumerate(categories)}
        values = [value_map[d] for d in hue]
    cmap = _norm_cmap(values, cmap, mpl.colors.Normalize, mpl.cm)
    return cmap, categories, values


def _paint_hue_legend(
    ax, categories, cmap, legend_labels, kwargs, color_kwarg, legend_kwargs, figure=False
):
    """
    Creates a discerete categorical legend for ``hue`` and attaches it to the axis.
    """
    markeredgecolor = kwargs['edgecolor'] if 'edgecolor' in kwargs else 'black'
    patches = []
    for value, _ in enumerate(categories):
        patches.append(
            mpl.lines.Line2D(
                [0], [0], linestyle='None',
                marker="o", markersize=10, markerfacecolor=cmap.to_rgba(value),
                markeredgecolor=markeredgecolor
            )
        )
    if not legend_kwargs:
        legend_kwargs = dict()
    legend_kwargs['frameon'] = legend_kwargs.pop('frameon', False)

    target = ax.figure if figure else ax
    if legend_labels:
        target.legend(patches, legend_labels, numpoints=1, fancybox=True, **legend_kwargs)
    else:
        target.legend(patches, categories, numpoints=1, fancybox=True, **legend_kwargs)



def _paint_colorbar_legend(ax, values, cmap, legend_kwargs):
    """
    Creates a continuous colorbar legend and attaches it to the axis.
    """
    if legend_kwargs is None:
        legend_kwargs = dict()
    legend_kwargs['frameon'] = legend_kwargs.pop('frameon', False)
    cmap.set_array(values)
    plt.gcf().colorbar(cmap, ax=ax, **legend_kwargs)


def _validate_buckets(df, hue, k, scheme):
    """
    This helper method infers if the ``hue`` parameter is categorical, and sets scheme if isn't
    already set.
    """
    if isinstance(hue, str):
        hue = df[hue]
    if hue is None:
        categorical = False
    # if the data is non-categorical, but there are fewer to equal numbers of bins and
    # observations, treat it as categorical, as doing so will make the legend cleaner
    elif k is not None and len(hue) <= k:
        categorical = True
    else:
        categorical = (hue.dtype == np.dtype('object'))
    scheme = scheme if scheme else 'Quantiles'
    return categorical, scheme


def _get_clip(extent, clip):
    xmin, ymin, xmax, ymax = extent
    # We have to add a little bit of padding to the edges of the box, as otherwise the edges
    # will invert a little, surprisingly.
    rect = shapely.geometry.Polygon(
        [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin), (xmin, ymin)]
    )
    rect = shapely.affinity.scale(rect, xfact=1.25, yfact=1.25)
    for geom in clip:
        rect = rect.symmetric_difference(geom)
    return rect


def _build_voronoi_polygons(df):
    """
    Given a GeoDataFrame of point geometries and pre-computed plot extrema, build Voronoi
    simplexes for the given points in the given space and returns them.

    Voronoi simplexes which are located on the edges of the graph may extend into infinity in some
    direction. In other words, the set of points nearest the given point does not necessarily have
    to be a closed polygon. We force these non-hermetic spaces into polygons using a subroutine.

    Returns a list of shapely.geometry.Polygon objects, each one a Voronoi polygon.
    """
    from scipy.spatial import Voronoi
    geom = np.array(df.geometry.map(lambda p: [p.x, p.y]).tolist())
    vor = Voronoi(geom)

    polygons = []

    for idx_point, _ in enumerate(vor.points):
        idx_point_region = vor.point_region[idx_point]
        idxs_vertices = np.array(vor.regions[idx_point_region])

        is_finite = not np.any(idxs_vertices == -1)

        if is_finite:
            # Easy case, the region is closed. Make a polygon out of the Voronoi ridge points.
            idx_point_region = vor.point_region[idx_point]
            idxs_vertices = np.array(vor.regions[idx_point_region])
            region_vertices = vor.vertices[idxs_vertices]
            region_poly = shapely.geometry.Polygon(region_vertices)

            polygons.append(region_poly)

        else:
            # Hard case, the region is open. Project new edges out to the margins of the plot.
            # See `scipy.spatial.voronoi_plot_2d` for the source of this calculation.
            point_idx_ridges_idx = np.where((vor.ridge_points == idx_point).any(axis=1))[0]

            # TODO: why does this happen?
            if len(point_idx_ridges_idx) == 0:
                continue

            ptp_bound = vor.points.ptp(axis=0)
            center = vor.points.mean(axis=0)

            finite_segments = []
            infinite_segments = []

            pointwise_ridge_points = vor.ridge_points[point_idx_ridges_idx]
            pointwise_ridge_vertices = np.asarray(vor.ridge_vertices)[point_idx_ridges_idx]

            for pointidx, simplex in zip(pointwise_ridge_points, pointwise_ridge_vertices):
                simplex = np.asarray(simplex)

                if np.all(simplex >= 0):
                    finite_segments.append(vor.vertices[simplex])

                else:
                    i = simplex[simplex >= 0][0]  # finite end Voronoi vertex

                    t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
                    t /= np.linalg.norm(t)
                    n = np.array([-t[1], t[0]])  # normal

                    midpoint = vor.points[pointidx].mean(axis=0)
                    direction = np.sign(np.dot(midpoint - center, n)) * n
                    far_point = vor.vertices[i] + direction * ptp_bound.max()

                    infinite_segments.append(np.asarray([vor.vertices[i], far_point]))

            finite_segments = finite_segments if finite_segments else np.zeros(shape=(0,2,2))
            ls = np.vstack([np.asarray(infinite_segments), np.asarray(finite_segments)])

            # We have to trivially sort the line segments into polygonal order. The algorithm that
            # follows is inefficient, being O(n^2), but "good enough" for this use-case.
            ls_sorted = []

            while len(ls_sorted) < len(ls):
                l1 = ls[0] if len(ls_sorted) == 0 else ls_sorted[-1]
                matches = []

                for l2 in [l for l in ls if not (l == l1).all()]:
                    if np.any(l1 == l2):
                        matches.append(l2)
                    elif np.any(l1 == l2[::-1]):
                        l2 = l2[::-1]
                        matches.append(l2)

                if len(ls_sorted) == 0:
                    ls_sorted.append(l1)

                for match in matches:
                    # in list sytax this would be "if match not in ls_sorted"
                    # in numpy things are more complicated...
                    if not any((match == ls_sort).all() for ls_sort in ls_sorted):
                        ls_sorted.append(match)
                        break

            # Build and return the final polygon.
            polyline = np.vstack(ls_sorted)
            geom = shapely.geometry.Polygon(polyline).convex_hull
            polygons.append(geom)

    return polygons


def _jitter_points(geoms):
    working_df = gpd.GeoDataFrame().assign(
        _x=geoms.x,
        _y=geoms.y,
        geometry=geoms
    )
    group = working_df.groupby(['_x', '_y'])
    group_sizes = group.size()

    if not (group_sizes > 1).any():
        return geoms

    else:
        jitter_indices = []

        group_indices = group.indices
        group_keys_of_interest = group_sizes[group_sizes > 1].index
        for group_key_of_interest in group_keys_of_interest:
            jitter_indices += group_indices[group_key_of_interest].tolist()

        _x_jitter = (
            pd.Series([0] * len(working_df)) +
            pd.Series(
                ((np.random.random(len(jitter_indices)) - 0.5)  * 10**(-5)),
                index=jitter_indices
            )
        )
        _x_jitter = _x_jitter.fillna(0)

        _y_jitter = (
            pd.Series([0] * len(working_df)) +
            pd.Series(
                ((np.random.random(len(jitter_indices)) - 0.5)  * 10**(-5)),
                index=jitter_indices
            )
        )
        _y_jitter = _y_jitter.fillna(0)

        out = gpd.GeoSeries([
            shapely.geometry.Point(x, y) for x, y in
            zip(working_df._x + _x_jitter, working_df._y + _y_jitter)
        ])

        # guarantee that no two points have the exact same coordinates
        regroup_sizes = (
            gpd.GeoDataFrame()
            .assign(_x=out.x, _y=out.y)
            .groupby(['_x', '_y'])
            .size()
        )
        assert not (regroup_sizes > 1).any()

        return out


#######################
# COMPATIBILITY SHIMS #
#######################

def _norm_cmap(values, cmap, normalize, cm):
    """
    Normalize and set colormap. Taken from geopandas@0.2.1 codebase, removed in geopandas@0.3.0.
    """
    mn = min(values)
    mx = max(values)
    norm = normalize(vmin=mn, vmax=mx)
    n_cmap = cm.ScalarMappable(norm=norm, cmap=cmap)
    return n_cmap
