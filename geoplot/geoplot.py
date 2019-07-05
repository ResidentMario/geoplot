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
import shapely.geometry
import pandas as pd
import descartes
from .ops import QuadTree, build_voronoi_polygons, jitter_points

try:
    from geopandas.plotting import _mapclassify_choro
except ImportError:
    from geopandas.plotting import __pysal_choro as _mapclassify_choro

__version__ = "0.3.0"


class HueMixin:
    """
    Class container for hue-setter code shared across all plots that support hue.
    """
    def set_hue_values(
        self, color_kwarg='color', default_color='steelblue',
        supports_continuous=True, supports_categorical=True,
        verify_input=True
    ):
        hue = self.kwargs.pop('hue', None)
        cmap = self.kwargs.pop('cmap', 'viridis')

        if supports_categorical:
            scheme = self.kwargs.pop('scheme')
            k = self.kwargs.pop('k')
        else:
            scheme = None
            k = None

        # kdeplot is special: it has a 'cmap' parameter but no 'hue' or 'scheme' parameters,
        # merely passing those parameters to `seaborn.kdeplot`. So for these plots we don't
        # validate.
        if verify_input:
            if color_kwarg in self.kwargs and hue is not None:
                raise ValueError(
                    f'Cannot specify both "{color_kwarg}" and "hue" in the same plot.'
                )
            # in theory we should also vet k, but since None is a meaningful value for this
            # variable it's not possible to know definitively if k was set as a default or by the
            # user.
            if (cmap is not None or scheme is not None) and hue is None:
                raise ValueError(
                    f'Cannot specify "cmap" or "scheme" without specifying "hue".'
                )

        hue = _to_geoseries(self.df, hue)
        if hue is None:  # no colormap
            color = self.kwargs.pop(color_kwarg, default_color)
            colors = [color] * len(self.df)
            categorical = False
            categories = None
        elif k is None:  # continuous colormap
            mn = min(hue)
            mx = max(hue)
            norm = mpl.colors.Normalize(vmin=mn, vmax=mx)
            cmap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            colors = [cmap.to_rgba(v) for v in hue]
            categorical = False
            categories = None
        else:  # categorical colormap
            categorical, scheme = _validate_buckets(self.df, hue, k, scheme)
            categories = None
            if hue is not None:
                # self.categorical is passed to ax.legend in paint_legend when legend_labels is
                # null and the legend is in hue mode (meaning hue is specified, legend is True, and
                # either legend_var is set to hue OR hue is the only valid legend variable in the
                # plot). ax.legend sorts the patch-label tuples by patch value in ascending order
                # internally. This obviates the need for sorting it ourselves here, but also forces
                # us to also use ascending (not descending) order on the scale legend also.
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
                colors = [cmap.to_rgba(v) for v in values]

        self.colors = colors
        self.hue = hue
        self.scheme = scheme
        self.k = k
        self.cmap = cmap
        self.categorical = categorical
        self.categories = categories
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
                # edge case: if dmax, dmin are <=10**-30 or so, will overflow and eval to infinity
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
            if self.scale_func is not None:
                raise ValueError(
                    f'Cannot specify "scale_func" without specifying "scale".'
                )
            size = self.kwargs.pop(size_kwarg, default_size)
            self.sizes = [size] * len(self.df)


class LegendMixin:
    """
    Class container for legend-builder code shared across all plots that support legend.
    """
    def paint_legend(
        self, supports_hue=True, supports_scale=False, verify_input=True, scale_multiplier=1,
    ):
        legend = self.kwargs.pop('legend', None)
        legend_labels = self.kwargs.pop('legend_labels', None)
        legend_values = self.kwargs.pop('legend_values', None)
        legend_kwargs = self.kwargs.pop('legend_kwargs', None)
        legend_marker_kwargs = dict()
        if legend_kwargs is None:
            legend_kwargs = dict()
        else:
            for kwarg in list(legend_kwargs.keys()):
                # as a power user feature, certain marker* parameters can be passed along to the
                # Line2D markers in the patch legend
                if kwarg[:6] == 'marker':
                    legend_marker_kwargs[kwarg] = legend_kwargs.pop(kwarg)

        # kdeplot is special: it has a 'legend' parameter but no other legend-related params,
        # as the 'legend' param is merely passed down to `seaborn.kdeplot`. So for kdeplot
        # we do not verify inputs.
        if verify_input:
            if legend and (
                (not supports_hue or self.hue is None) and
                (not supports_scale or self.scale is None)
            ):
                raise ValueError(
                    '"legend" is set to True, but the plot has neither a "hue" nor a "scale" '
                    'variable.'
                )
            if not legend and (
                legend_labels is not None or legend_values is not None or
                legend_kwargs != dict()
            ):
                raise ValueError(
                    'Cannot specify "legend_labels", "legend_values", or "legend_kwargs" '
                    'when "legend" is set to False.'
                )
            if (
                legend_labels is not None and legend_values is not None and
                len(legend_labels) != len(legend_values)
            ):
                raise ValueError(
                    'The "legend_labels" and "legend_values" parameters have diferent lengths.'
                )

        # Mutate matplotlib defaults
        addtl_legend_kwargs = dict()
        addtl_legend_kwargs['fancybox'] = legend_kwargs.pop('fancybox', True)

        if supports_hue and supports_scale:
            if self.kwargs['legend_var'] is not None:
                legend_var = self.kwargs['legend_var']
                if legend_var not in ['scale', 'hue']:
                    raise ValueError(
                        f'"legend_var", if specified, must be set to one of "hue" or "scale".'
                    )
                elif getattr(self, legend_var) is None:
                    raise ValueError(
                        f'"legend_var" is set to "{legend_var!r}", but "{legend_var!r}" is '
                        f'unspecified.'
                    )
            else:
                if self.hue is not None and self.scale is not None:
                    warnings.warn(
                        f'Please specify "legend_var" explicitly when both "hue" and "scale" are '
                        f'specified. Defaulting to "legend_var=\'hue\'".'
                    )
                legend_var = 'hue' if self.hue is not None else 'scale'
            self.kwargs.pop('legend_var')
        else:
            legend_var = 'hue'

        if legend and legend_var == 'hue':
            if self.k is not None:
                # If the user provides a markeredgecolor in legend_kwargs, use that. Otherwise,
                # if they provide an edgecolor in kwargs, reuse that. Otherwise, default to a
                # transparent markeredgecolor.
                if 'markeredgecolor' in legend_marker_kwargs:
                    markeredgecolor = legend_marker_kwargs.pop('markeredgecolor')
                elif 'edgecolor' in self.kwargs:
                    markeredgecolor = self.kwargs.get('edgecolor')
                else:
                    markeredgecolor = 'None'

                # If the user provides a markerfacecolor in legend_kwargs, but the legend is in
                # hue mode, raise an error, as setting this markerfacecolor would invalidate the
                # utility of the legend.
                if 'markerfacecolor' in legend_marker_kwargs:
                    raise ValueError(
                        f'Cannot set a "markerfacecolor" when the "legend_var" is set to "hue". '
                        f'Doing so would remove the color reference, rendering the legend '
                        f'useless. Are you sure you didn\'t mean to set "markeredgecolor" instead?'
                    )

                marker_kwargs = {
                    'marker': "o", 'markersize': 10, 'markeredgecolor': markeredgecolor
                }
                marker_kwargs.update(legend_marker_kwargs)

                patches = []
                for value, _ in enumerate(self.categories):
                    patches.append(
                        mpl.lines.Line2D(
                            [0], [0], linestyle='None',
                            markerfacecolor=self.cmap.to_rgba(value),
                            **marker_kwargs
                        )
                    )
                if legend_labels:
                    if len(patches) != len(legend_labels):
                        raise ValueError(
                            f'The list of legend values is length {len(patches)}, but the list of '
                            f'legend_labels is length {len(legend_labels)}.'
                        )
                    self.ax.legend(
                        patches, legend_labels, numpoints=1,
                        **legend_kwargs, **addtl_legend_kwargs
                    )
                else:
                    self.ax.legend(
                        patches, self.categories, numpoints=1,
                        **legend_kwargs, **addtl_legend_kwargs
                    )
            else:  # self.k is None
                if len(legend_marker_kwargs) > 0:
                    raise ValueError(
                        f'"k" is set to "None", implying a colorbar legend, but "legend_kwargs" '
                        f'includes marker parameters that can only be applied to a patch legend. '
                        f'Remove these parameters or convert to a categorical colormap by '
                        f'specifying a "k" value.'
                    )

                if legend_labels is not None or legend_values is not None:
                    # TODO: implement this feature
                    raise NotImplementedError(
                        f'"k" is set to "None", implying a colorbar legend, but "legend_labels" '
                        f'and/or "legend_values" are also specified. These parameters do not '
                        f'apply in the case of a colorbar legend and should be removed.'
                    )

                self.cmap.set_array(self.hue)
                plt.gcf().colorbar(self.cmap, ax=self.ax, **legend_kwargs)

        elif legend and legend_var == 'scale':
            if legend_values is None:
                # If the user doesn't specify their own legend_values, apply a reasonable
                # default: a five-point linear array from max to min. The sort order (max to min,
                # not min to max) is important because ax.legend, the matplotlib function these
                # values are ultimately passed to, sorts the patches in ascending value order
                # internally. Even though we pass no patch ordering information to ax.legend,
                # it still appears to determine an ordering by inspecting plot artists
                # automagically. In the case where there is no colormap, however, the patch order
                # we pass is preserved.
                #
                # The TLDR is that it's possible to control scale legend patch order (and make it
                # ascending or descending) in a non-hue plot, in all other cases legend patch order
                # is locked to ascending, so for consistency across the API we use ascending order
                # in this case as well.
                legend_values = np.linspace(
                    np.max(self.scale), np.min(self.scale), num=5, dtype=self.scale.dtype
                )[::-1]
            if legend_labels is None:
                # If the user doesn't specify their own legend_labels, apply a reasonable
                # default: the 'g' f-string for the given input value.
                legend_labels = ['{0:g}'.format(value) for value in legend_values]

            # If the user specifies a markerfacecolor explicity via legend_params, use that.
            #
            # Otherwise, use an open-circle design when hue is not None, so as not to confuse
            # viewers with colors in the scale mapping to values that do not correspond with the
            # plot points. But if there is no hue, it's better to have the legend markers be as
            # close to the plot markers as possible, so in that case the points are filled-in with
            # the corresponding plot color value. This is controlled by self.colors and, in the
            # case where hue is None, will be an n-length list of the same color value or name, so
            # we can grab that by taking the first element of self.colors.
            if 'markerfacecolor' in legend_marker_kwargs:
                markerfacecolors = [legend_marker_kwargs['markerfacecolor']] * len(legend_values)
                legend_marker_kwargs.pop('markerfacecolor')
            elif self.hue is None:
                markerfacecolors = [self.colors[0]] * len(legend_values)
            else:
                markerfacecolors = ['None'] * len(legend_values)

            markersizes = [self.dscale(d) * scale_multiplier for d in legend_values]

            # If the user provides a markeredgecolor in legend_kwargs, use that. Otherwise, default
            # to a steelblue or black markeredgecolor, depending on whether hue is defined.
            if 'markeredgecolor' in legend_marker_kwargs:
                markeredgecolor = legend_marker_kwargs.pop('markeredgecolor')
            elif self.hue is None:
                markeredgecolor = 'steelblue'
            else:
                markeredgecolor = 'black'

            # If the user provides a markersize in legend_kwargs, but the legend is in
            # scale mode, raise an error, as setting this markersize would invalidate the
            # utility of the legend.
            if 'markersize' in legend_marker_kwargs:
                raise ValueError(
                    f'Cannot set a "markersize" when the "legend_var" is set to "scale". '
                    f'Doing so would remove the scale reference, rendering the legend '
                    f'useless.'
                )

            marker_kwargs = {
                'marker': "o", 'markeredgecolor': markeredgecolor
            }
            marker_kwargs.update(legend_marker_kwargs)

            patches = []
            for markerfacecolor, markersize in zip(
                markerfacecolors, markersizes
            ):
                patches.append(
                    mpl.lines.Line2D(
                        [0], [0], linestyle='None',
                        markersize=markersize,
                        markerfacecolor=markerfacecolor,
                        **marker_kwargs
                    )
                )

            if len(patches) != len(legend_labels):
                raise ValueError(
                    f'The list of legend values is length {len(patches)}, but the list of '
                    f'legend_labels is length {len(legend_labels)}.'
                )

            self.ax.legend(
                patches, legend_labels, numpoints=1, **legend_kwargs, **addtl_legend_kwargs
            )


class ClipMixin:
    """
    Class container for clip-setter code shared across all plots that support clip.

    Note that there are two different routines for clipping a plot:
    * Drawing an inverted polyplot as the top layer. Implemented in `paint_clip`. Advantage is
      that it is fast, disadvantage is that the resulting plot can't be applied to a webmap.
    * Intersecting each geometry with the unary union of the clip geometries. This is a slower
      but more broadly compatible process. It's also quite fast if the clip geometry used is
      relatively simple, but this requires conscious effort the user (we can't simplify
      automatically unfortunately).

    KDEPlot uses the first method because it relies on `seaborn` underneath, and there is no way
    to clip an existing axis painter (that I am aware of). All other plots use the second method.
    """
    def set_clip(self, gdf):
        clip = self.kwargs.pop('clip')
        clip = _to_geoseries(gdf, clip)

        if clip is not None:
            clip_shp = clip.unary_union
            gdf = gdf.assign(
                geometry=gdf.geometry.intersection(clip_shp)
            )
        return gdf

    @staticmethod
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

    def paint_clip(self):
        clip = self.kwargs.pop('clip')
        clip = _to_geoseries(self.df, clip)
        if clip is not None:
            if self.projection is not None:
                clip_geom = self._get_clip(self.ax.get_extent(crs=ccrs.PlateCarree()), clip)
                feature = ShapelyFeature([clip_geom], ccrs.PlateCarree())
                self.ax.add_feature(feature, facecolor=(1,1,1), linewidth=0, zorder=2)
            else:
                clip_geom = self._get_clip(self.ax.get_xlim() + self.ax.get_ylim(), clip)
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
        # exceeds 1cm, jitter the points yourself. cf. https://xkcd.com/2170/
        df = df.assign(geometry=jitter_points(df.geometry))

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
        nsig = self.kwargs.pop('nsig')
        _df = self.df
        dvals = []

        # If no hue is set, getting the correct (null) colormap values is as easy as calling
        # the same set_hue_values used by most other plots.
        #
        # If hue *is* set, things are more complicated. The quadtree colormap is applied to a map
        # over self.partitions, but set_hue_values is called on self.df. So we temporarily swap
        # self.df out for the map on self.partitions, run set_hue_values, then swap the original
        # GeoDataFrame back into place. We apply the nsig adjustment afterwards.
        has_hue = 'hue' in self.kwargs and self.kwargs['hue'] is not None
        if has_hue:
            for p in self.partitions:
                if len(p.data) == 0:  # empty
                    dval = agg(pd.Series([0]))
                elif has_hue:
                    dval = agg(p.data.hue_col)
                dvals.append(dval)

            self.df = pd.DataFrame({
                self.kwargs['hue']: dvals
            })
            super().set_hue_values(color_kwarg='facecolor', default_color='None')
            self.df = _df

            # apply the special nsig parameter colormap rule
            for i, dval in enumerate(dvals):
                if dval < nsig:
                    self.colors[i] = 'None'
        else:
            super().set_hue_values(color_kwarg='facecolor', default_color='None')


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
            xmin, ymin, xmax, ymax = self.df.total_bounds
            # Plots suffer clipping issues if we use just the geometry extrema due to plot features
            # that fall outside of the viewport. The most common case is the edges of polygon
            # patches with non-zero linewidth. We partially ameliorate the problem by increasing
            # the viewport by 1% of the total coordinate area of the plot. Note that the
            # coordinate area covered will differ from the actual area covered, as distance between
            # degrees varies depending on where you are on the globe. Since the effect is small we
            # ignore this problem here, for simplicity's sake.
            viewport_area = (xmax - xmin) * (ymax - ymin)
            window_resize_val = 0.0025 * viewport_area
            extrema = np.array([
                np.max([-180, xmin - window_resize_val]),
                np.max([-90, ymin - window_resize_val]),
                np.min([180, xmax + window_resize_val]),
                np.min([90, ymax + window_resize_val])
            ])

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

            if self.ax is None:
                ax = plt.subplot(111, projection=self.projection)
            else:
                ax = self.ax

        else:
            if self.ax is None:
                ax = plt.gca()
            else:
                ax = self.ax

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
                    # The default behavior in cartopy is to use a global extent with
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
    hue=None, cmap=None, k=5, scheme=None,
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
        `Working with Projections
        <https://residentmario.github.io/geoplot/user_guide/Working_with_Projections.html>`_.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#hue>`_.
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
        `Customizing Plots#Scale 
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#scale>`_.
    limits : (min, max) tuple, optional
        If ``scale`` is set, the minimum and maximum size of the points.
    scale_func : ufunc, optional
        If ``scale`` is set, the function used to determine the size of each point. For reference
        see the
        `Pointplot Scale Functions
        <https://residentmario.github.io/geoplot/gallery/plot_usa_city_elevations.html#sphx-glr-gallery-plot-usa-city-elevations-py>`_
        demo.
    legend : boolean, optional
        Whether or not to include a map legend. For a reference on this and the other 
        legend-related parameters that follow, see
        `Customizing Plots#Legend
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_var : "hue" or "scale", optional
        Which variable, ``hue`` or ``scale``, to use in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the underlying legend.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#extent>`_.
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
    """
    class PointPlot(Plot, HueMixin, ScaleMixin, LegendMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.set_hue_values(color_kwarg='color', default_color='steelblue')
            self.set_scale_values(size_kwarg='s', default_size=5)
            self.paint_legend(supports_hue=True, supports_scale=True)

        def draw(self):
            ax = plot.ax
            if len(plot.df.geometry) == 0:
                return ax

            xs = np.array([p.x for p in plot.df.geometry])
            ys = np.array([p.y for p in plot.df.geometry])
            if self.projection:
                ax.scatter(
                    xs, ys, transform=ccrs.PlateCarree(), c=plot.colors,
                    # the ax.scatter 's' param is an area but the API is unified on width in pixels
                    # (or "points"), so we have to square the value at draw time to get the correct
                    # point size.
                    s=[s**2 for s in plot.sizes],
                    **plot.kwargs
                )
            else:
                ax.scatter(xs, ys, c=plot.colors, s=plot.sizes, **plot.kwargs)
            return ax

    plot = PointPlot(
        df, figsize=figsize, ax=ax, extent=extent, projection=projection,
        hue=hue, scheme=scheme, k=k, cmap=cmap,
        scale=scale, limits=limits, scale_func=scale_func,
        legend=legend, legend_var=legend_var, legend_values=legend_values,
        legend_labels=legend_labels, legend_kwargs=legend_kwargs,
        **kwargs
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
        `Working with Projections
        <https://residentmario.github.io/geoplot/user_guide/Working_with_Projections.html>`_.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#extent>`_.
    figsize : (x, y) tuple, optional
        Sets the size of the plot figure (in inches).
    ax : AxesSubplot or GeoAxesSubplot instance, optional
        If set, the ``matplotlib.axes.AxesSubplot`` or ``cartopy.mpl.geoaxes.GeoAxesSubplot``
        instance to paint the plot on. Defaults to a new axis.
    kwargs: dict, optional
        Keyword arguments to be passed to the underlying matplotlib `Polygon patches
        <http://matplotlib.org/api/patches_api.html#matplotlib.patches.Polygon>`_.

    Returns
    -------
    ``AxesSubplot`` or ``GeoAxesSubplot``
        The plot axis.
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
    hue=None, cmap=None, k=5, scheme=None,
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
        `Working with Projections
        <https://residentmario.github.io/geoplot/user_guide/Working_with_Projections.html>`_.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#hue>`_.
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
        `Customizing Plots#Legend
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the underlying legend.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#extent>`_.
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
    hue=None, cmap=None, k=5, scheme=None,
    nmax=None, nmin=None, nsig=0, agg=np.mean,
    legend=False, legend_kwargs=None, legend_values=None, legend_labels=None,
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
        `Working with Projections
        <https://residentmario.github.io/geoplot/user_guide/Working_with_Projections.html>`_.
    clip : None or iterable or GeoSeries, optional
        If specified, quadrangles will be clipped to the boundaries of this geometry.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#hue>`_.
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
        `Customizing Plots#Legend
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the underlying legend.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#extent>`_.
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
    """
    class QuadtreePlot(Plot, QuadtreeComputeMixin, QuadtreeHueMixin, LegendMixin, ClipMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.compute_quadtree()
            self.set_hue_values(color_kwarg='facecolor', default_color='None')
            self.paint_legend(supports_hue=True, supports_scale=False)

        def draw(self):
            ax = self.ax
            if len(self.df.geometry) == 0:
                return ax

            geoms = []
            for p in self.partitions:
                xmin, xmax, ymin, ymax = p.bounds
                rect = shapely.geometry.Polygon(
                    [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)]
                )
                geoms.append(rect)
            geoms = gpd.GeoDataFrame(geometry=geoms)
            geoms = self.set_clip(geoms)

            for geom, color in zip(geoms.geometry, self.colors):
                # Splitting rules that specify an nmax but not an nmin can result in partitions
                # which are completely outside (e.g. do not intersect at all with) the clip
                # geometry. The intersection operation run in set_clip will return an empty
                # GeometryCollection for these results. The plot drivers do not try to interpret
                # GeometryCollection objects, even empty ones, and will raise an error when passed
                # one, so we have to exclude these bad partitions ourselves.
                if (
                    isinstance(geom, shapely.geometry.GeometryCollection) and
                    len(geom) == 0
                ):
                    continue

                if projection:
                    feature = ShapelyFeature([geom], ccrs.PlateCarree())
                    ax.add_feature(
                        feature, facecolor=color, **self.kwargs
                    )
                else:
                    feature = descartes.PolygonPatch(
                        geom, facecolor=color, **self.kwargs
                    )
                    ax.add_patch(feature)

            return ax

    plot = QuadtreePlot(
        df, projection=projection,
        clip=clip,
        hue=hue, scheme=scheme, k=k, cmap=cmap,
        nmax=nmax, nmin=nmin, nsig=nsig, agg=agg,
        legend=legend, legend_values=legend_values, legend_labels=legend_labels,
        legend_kwargs=legend_kwargs,
        extent=extent, figsize=figsize, ax=ax,
        **kwargs
    )
    return plot.draw()


def cartogram(
    df, projection=None,
    scale=None, limits=(0.2, 1), scale_func=None,
    hue=None, cmap=None, k=5, scheme=None,
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
        `Working with Projections
        <https://residentmario.github.io/geoplot/user_guide/Working_with_Projections.html>`_.
    scale : str or iterable, optional
        The column in the dataset (or an iterable of some other data) with which to scale output
        points. For a reference on this and the other scale-related parameters that follow, see
        `Customizing Plots#Scale 
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#scale>`_.
    limits : (min, max) tuple, optional
        If ``scale`` is set, the minimum and maximum size of the points.
    scale_func : ufunc, optional
        If ``scale`` is set, the function used to determine the size of each point. For reference
        see the
        `Pointplot Scale Functions
        <https://residentmario.github.io/geoplot/gallery/plot_usa_city_elevations.html#sphx-glr-gallery-plot-usa-city-elevations-py>`_
        demo.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#hue>`_.
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
        `Customizing Plots#Legend
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_var : "hue" or "scale", optional
        Which variable, ``hue`` or ``scale``, to use in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the underlying legend.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#extent>`_.
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
    """
    if scale is None:
        raise ValueError("No scale parameter provided.")

    class CartogramPlot(Plot, HueMixin, ScaleMixin, LegendMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.set_scale_values(size_kwarg=None, default_size=None)
            self.set_hue_values(color_kwarg='facecolor', default_color='steelblue')

            # Scaling a legend marker means scaling a point, whereas scaling a cartogram
            # marker means scaling a polygon. The same scale has radically different effects
            # on these two in perceptive terms. The scale_multiplier helps to make the point
            # scaling commutative to the polygon scaling, though it's really just a guess.
            # 25 is chosen because it is 5**2, where 5 is a "good" value for the radius of a
            # point in a scatter point.
            self.paint_legend(
                supports_hue=True, supports_scale=True, scale_multiplier=25
            )

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
        `Working with Projections
        <https://residentmario.github.io/geoplot/user_guide/Working_with_Projections.html>`_.
    cmap : matplotlib color, optional
        The `colormap <http://matplotlib.org/examples/color/colormaps_reference.html>`_ to use.
    clip : None or iterable or GeoSeries, optional
        If specified, isochrones will be clipped to the boundaries of this geometry.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#extent>`_.
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
    """
    import seaborn as sns  # Immediately fail if no seaborn.

    class KDEPlot(Plot, HueMixin, LegendMixin, ClipMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.set_hue_values(
                color_kwarg=None, default_color=None, supports_categorical=False, verify_input=False
            )
            self.paint_legend(supports_hue=True, supports_scale=False, verify_input=False)
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
        shade_lowest=shade_lowest,
        **kwargs
    )
    return plot.draw()


def sankey(
    df, projection=None,
    hue=None, cmap=None, k=5, scheme=None,
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
        `Working with Projections
        <https://residentmario.github.io/geoplot/user_guide/Working_with_Projections.html>`_.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#hue>`_.
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
        `Customizing Plots#Scale 
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#scale>`_.
    limits : (min, max) tuple, optional
        If ``scale`` is set, the minimum and maximum size of the points.
    scale_func : ufunc, optional
        If ``scale`` is set, the function used to determine the size of each point. For reference
        see the
        `Pointplot Scale Functions
        <https://residentmario.github.io/geoplot/gallery/plot_usa_city_elevations.html#sphx-glr-gallery-plot-usa-city-elevations-py>`_
        demo.
    legend : boolean, optional
        Whether or not to include a map legend. For a reference on this and the other 
        legend-related parameters that follow, see
        `Customizing Plots#Legend
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_var : "hue" or "scale", optional
        Which variable, ``hue`` or ``scale``, to use in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the underlying legend.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#extent>`_.
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
    """
    class SankeyPlot(Plot, HueMixin, ScaleMixin, LegendMixin):
        def __init__(self, df, **kwargs):
            # Most markers use 'color' or 'facecolor' as their color_kwarg, and this parameter
            # has the same name as a matplotlib feature (for unprojected plots) and as a decartes
            # feature (for projected plots). With line markers, things are different. The
            # matplotlib Line2D marker uses 'color' (it also has an 'markeredgecolor', but this
            # parameter doesn't perform exactly like the 'edgecolor' elsewhere in the API). The
            # descartes feature uses 'edgecolor'.
            # 
            # This complicates keywords in the Sankey API (the only plot type so far that uses a
            # line marker). For code cleanliness, we choose to support "color" and not
            # "edgecolor", and to raise if an "edgecolor" is set.
            if 'edgecolor' in kwargs:
                raise ValueError(
                    f'Invalid parameter "edgecolor". To control line color, use "color".'
                )

            super().__init__(df, **kwargs)
            self.set_hue_values(color_kwarg='color', default_color='steelblue')
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
    hue=None, cmap=None, k=5, scheme=None,
    legend=False, legend_kwargs=None, legend_labels=None, legend_values=None,
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
        `Working with Projections
        <https://residentmario.github.io/geoplot/user_guide/Working_with_Projections.html>`_.
    clip : None or iterable or GeoSeries, optional
        If specified, the output will be clipped to the boundaries of this geometry.
    hue : None, Series, GeoSeries, iterable, or str, optional
        The column in the dataset (or an iterable of some other data) used to color the points.
        For a reference on this and the other hue-related parameters that follow, see
        `Customizing Plots#Hue
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#hue>`_.
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
        `Customizing Plots#Scale 
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#scale>`_.
    limits : (min, max) tuple, optional
        If ``scale`` is set, the minimum and maximum size of the points.
    scale_func : ufunc, optional
        If ``scale`` is set, the function used to determine the size of each point. For reference
        see the
        `Pointplot Scale Functions
        <https://residentmario.github.io/geoplot/gallery/plot_usa_city_elevations.html#sphx-glr-gallery-plot-usa-city-elevations-py>`_
        demo.
    legend : boolean, optional
        Whether or not to include a map legend. For a reference on this and the other 
        legend-related parameters that follow, see
        `Customizing Plots#Legend
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#legend>`_.
    legend_values : list, optional
        The data values to be used in the legend.
    legend_labels : list, optional
        The data labels to be used in the legend.
    legend_var : "hue" or "scale", optional
        Which variable, ``hue`` or ``scale``, to use in the legend.
    legend_kwargs : dict, optional
        Keyword arguments to be passed to the underlying legend.
    extent : None or (min_longitude, min_latitude, max_longitude, max_latitude), optional
        Controls the plot extents. For reference see 
        `Customizing Plots#Extent
        <https://residentmario.github.io/geoplot/user_guide/Customizing_Plots.html#extent>`_.
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
    """
    class VoronoiPlot(Plot, HueMixin, LegendMixin, ClipMixin):
        def __init__(self, df, **kwargs):
            super().__init__(df, **kwargs)
            self.set_hue_values(color_kwarg='facecolor', default_color='None')
            self.paint_legend(supports_hue=True, supports_scale=False)

        def draw(self):
            ax = self.ax
            if len(df.geometry) == 0:
                return ax

            geoms = build_voronoi_polygons(self.df)
            # Must take the .values of the output GeoDataFrame because assign is index-aligned.
            # So self.df can have any index. But set_clip constructs a new GeoSeries with a fresh
            # descending (1..N) index. If self.df doesn't also have a 1..N index, the join will
            # be misaligned and/or nan values will be inserted. The easiest way to assign in an
            # index-naive (e.g. index-based) manner is to provide a numpy array instead of a
            # GeoSeries by taking .values.
            self.df = self.df.assign(
                geometry=self.set_clip(gpd.GeoDataFrame(geometry=geoms)).values
            )
            if self.projection:
                for color, geom in zip(self.colors, self.df.geometry):
                    features = ShapelyFeature([geom], ccrs.PlateCarree())
                    ax.add_feature(features, facecolor=color, edgecolor=edgecolor, **self.kwargs)
            else:
                for color, geom in zip(self.colors, self.df.geometry):
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
