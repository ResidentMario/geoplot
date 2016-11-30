import geopandas as gpd
from geopandas.plotting import __pysal_choro, norm_cmap
import pandas as pd
from shapely import geometry
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm
from matplotlib.lines import Line2D
import numpy as np
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs
import warnings
from collections import defaultdict


def pointplot(df,
              extent=None,
              hue=None,
              scheme=None, k=None, cmap='Set1', categorical=False, vmin=None, vmax=None,
              stock_image=False, coastlines=False, gridlines=False,
              projection=None,
              legend=False, legend_kwargs=None,
              figsize=(8, 6),
              **kwargs):
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
        ax.set_extent(extent)

    # Set optional parameters.
    _set_optional_parameters(ax, stock_image, coastlines, gridlines)

    # Clean up patches.
    _lay_out_axes(ax)

    # Set up the colormap. This code is largely taken from geoplot's choropleth facilities, cf.
    # https://github.com/geopandas/geopandas/blob/master/geopandas/plotting.py#L253
    # If a scheme is provided we compute a distribution for the given data. If one is not provided we assume that the
    # input data is categorical.
    if hue is not None:
        cmap, categories, values = _colorize(categorical, hue, scheme, k, cmap, vmin, vmax)
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
               hue=None,
               scheme=None, k=None, cmap='Set1', categorical=False, vmin=None, vmax=None,
               legend=False, legend_kwargs=None,
               extent=None,
               stock_image=False, coastlines=False, gridlines=False,
               projection=None,
               figsize=(8, 6),
               **kwargs):

    # Format the data to be displayed for input.
    if not hue:
        nongeom = set(df.columns) - {df.geometry.name}
        if len(nongeom) > 1:
            raise ValueError("Ambiguous input: no 'hue' parameter was specified and the inputted DataFrame has more "
                             "than one column of data.")
        else:
            hue = df[list(nongeom)[0]]
    elif isinstance(hue, str):
        hue = df[hue]

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
        ax.set_extent(extent)
    else:
        ax.set_extent((x_min_coord, x_max_coord, y_min_coord, y_max_coord))

    # Set optional parameters.
    _set_optional_parameters(ax, stock_image, coastlines, gridlines)

    # Generate colormaps.
    cmap, categories, values = _colorize(categorical, hue, scheme, k, cmap, vmin, vmax)

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
            cutoff=5,
            hue=None,
            # scheme=None, k=None, cmap='Set1', categorical=False, vmin=None, vmax=None,
            # legend=False, legend_kwargs=None,
            extent=None,
            # stock_image=False, coastlines=False, gridlines=False,
            projection=None,
            figsize=(8, 6),
            **kwargs):
    # TODO: Implement the missingno geographic nullity plot as a general-purpose plot type here.
    # We need to smartly generate geometry limits. For now we'll just use centroids.
    xs = [g.x for g in df.geometry]
    ys = [g.y for g in df.geometry]
    minx, maxx = np.min(xs), np.max(xs)
    miny, maxy = np.min(ys), np.max(ys)
    patches = dict()
    # Do stuff.
    import pdb; pdb.set_trace()
    patches = _squarify(df, (minx, maxx, miny, maxy), cutoff)
    1 + 1
    1 + 1



##################
# HELPER METHODS #
##################


def _get_envelopes_min_maxes(envelopes):
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
    xmin, xmax, ymin, ymax = _get_envelopes_min_maxes(envelopes)
    return np.mean(xmin, xmax), np.mean(ymin, ymax)


def _lay_out_axes(ax):
    # Enabled by default is a transparent background patch and an "outline" patch that forms a border.
    # This code removes the extraneous patches.
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)


def _colorize(categorical, hue, scheme, k, cmap, vmin, vmax):
    # Set up the colormap. This code is largely taken from geoplot's choropleth facilities, cf.
    # https://github.com/geopandas/geopandas/blob/master/geopandas/plotting.py#L253
    # If a scheme is provided we compute a distribution for the given data. If one is not provided we assume that the
    # input data is categorical.
    # TODO: The "scheme" specification used by geoplot is inconsistent, consider fixing that.
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
    if stock_image:
        ax.stock_img()
    if coastlines:
        ax.coastlines()
    if gridlines:
        ax.gridlines()


def _validate_buckets(categorical, k, scheme):
    # Validate hue bucketing input. Valid inputs are:
    # 1. Both k and scheme are specified. In that case the user wants us to handle binning the data into k buckets
    #    ourselves, using the stated algorithm. We issue a warning if the specified k is greater than 10.
    # 2. k is left unspecified and scheme is specified. In that case the user wants us to handle binning the data
    #    into some default (k=5) number of buckets, using the stated algorithm.
    # 3. Both k and scheme are left unspecified. In that case the user wants us bucket the data variable using some
    #    default algorithm (Quantiles) into some default number of buckets (5).
    # 4. k is specified, but scheme is not. We choose to interpret this as meaning that the user wants us to handle
    #    bucketing the data into k buckets using the default (Quantiles) bucketing algorithm.
    # 5. categorical is True, and both k and scheme are False or left unspecified. In that case we do categorical.
    # Invalid inputs are:
    # 6. categorical is True, and one of k or scheme are also specified. In this case we raise a ValueError as this
    #    input makes no sense.
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


def _generate_patches(df, window, cutoff):
    quads = _squarify(df, window, cutoff)
    if len(quads) <= 4:
        return quads
    else:
        for i in range(len(quads) - 4):
            # cleanup...
            pass


def _squarify(df, window, cutoff):
    # TODO: Write this as a tree structure.
    min_x, max_x, min_y, max_y = window
    indices = __indices_inside(df, window)
    threshold = cutoff * len(df) if cutoff < 1 else cutoff  # both float (percentage) and integer cutoffs allowed
    if len(indices) > threshold:
        mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        q1 = _squarify(df, (min_x, mid_x, mid_y, max_y), cutoff)
        q2 = _squarify(df, (min_x, mid_x, min_y, mid_y), cutoff)
        q3 = _squarify(df, (mid_x, max_x, mid_y, max_y), cutoff)
        q4 = _squarify(df, (mid_x, max_x, min_y, mid_y), cutoff)
        return [((min_x, max_x, min_y, max_y), indices)] + q1 + q2 + q3 + q4
        # mid_x, mid_y = (min_x + max_x) / 2, (min_y + max_y) / 2
        # q1w = (min_x, mid_x, mid_y, max_y)
        # q2w = (min_x, mid_x, min_y, mid_y)
        # q3w = (mid_x, max_x, mid_y, max_y)
        # q4w = (mid_x, max_x, min_y, mid_y)
        # q1i = __indices_inside(df, q1w)
        # q2i = __indices_inside(df, q2w)
        # q3i = __indices_inside(df, q3w)
        # q4i = __indices_inside(df, q4w)
        # subl = []
        # for qw, qi in zip([q1w, q2w, q3w, q4w], [q1i, q2i, q3i, q4i]):
        #     if len(qi) > threshold:
        #         subl += _squarify(df.iloc[qi], qw, cutoff)
        # if len(subl) > 0:
        #     return subl
        # else:
        #     return [[window, indices]]
    else:
        # return [[window, indices]]
        return []


def __indices_inside(df, window):
    min_x, max_x, min_y, max_y = window
    points = df.geometry.centroid
    is_in = points.map(lambda point: (min_x < point.x < max_x) & (min_y < point.y < max_y))
    indices = is_in.values.nonzero()[0]
    return indices

# points_inside = df[(_min_x < arr[:, 0]) &
    #                    (arr[:, 0] < _max_x) &
    #                    (_min_y < arr[:, 1]) &
    #                    (arr[:, 1] < _max_y)]
    # if len(points_inside) < cutoff:
    #     # The following subroutine groups `geo_group` by `x_col` and `y_col`, and calculates and returns
    #     # a list of points in the group (`points`) as well as its overall nullity (`geographic_nullity`). The
    #     # first of these calculations is ignored.
    #     _, square_nullity = _calculate_geographic_nullity(points_inside, x_col, y_col)
    #     rectangles.append(((_min_x, _max_x, _min_y, _max_y), square_nullity))
    # else:
    #     _mid_x, _mid_y = (_min_x + _max_x) / 2, (_min_y + _max_y) / 2
    #     squarify(_min_x, _mid_x, _mid_y, _max_y, points_inside)
    #     squarify(_min_x, _mid_x, _min_y, _mid_y, points_inside)
    #     squarify(_mid_x, _max_x, _mid_y, _max_y, points_inside)
    #     squarify(_mid_x, _max_x, _min_y, _mid_y, points_inside)
