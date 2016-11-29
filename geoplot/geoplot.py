import geopandas as gpd
from geopandas.plotting import __pysal_choro, norm_cmap
import pandas as pd
from shapely import geometry
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm
import numpy as np
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs


def pointplot(df,
              extent=None,
              stock_image=False, coastlines=False,
              projection=None,
              figsize=(8, 6),
              **kwargs):
    # Initialize the figure.
    fig = plt.figure(figsize=figsize)

    # TODO
    # # If we are not handed a projection we are in the PateCarree projection. In that case we can return a
    # # `matplotlib` plot directly, which has the advantage of being native to e.g. mplleaflet.
    # if not projection:
    #     xs = np.array([p.x for p in df.geometry])
    #     ys = np.array([p.y for p in df.geometry])
    #     return plt.scatter(xs, ys)
    # Otherwise, we have to deal with projection settings.

    # All of the optional parameters passed to the Cartopy CRS instance as an argument to the projection parameter
    # above are themselves passed on, via initialization, into a `proj4_params` attribute of the class, which is a
    # basic dict of the form e.g.:
    # >>> projection.proj4_params
    # <<< {'proj': 'eqc', 'lon_0': 0.0, 'a': 57.29577951308232, 'ellps': 'WGS84'}
    #
    # In general Python follows the philosophy that everything should be mutable. This object, however,
    # refuses assignment. For example witness what happens when you insert the following code:
    # >>> projection.proj4_params['a'] = 0
    # >>> print(projection.proj4_params['a'])
    # <<< 57.29577951308232
    # In other words, Cartopy CRS internals are immutable; they can only be set at initialization.
    # cf. http://stackoverflow.com/questions/40822241/seemingly-immutable-dict-in-object-instance/40822473
    #
    # I tried several workarounds. The one which works best is having the user pass a geoplot.crs.* to projection;
    # the contents of geoplot.crs are a bunch of thin projection class wrappers with a factory method, "load",
    # for properly configuring a Cartopy projection with or without optional central coordinate(s).
    projection = projection.load(df, {
        'central_longitude': lambda df: np.mean(np.array([p.x for p in df.geometry.centroid])),
        'central_latitude': lambda df: np.mean(np.array([p.y for p in df.geometry.centroid]))
    })

    # Set up the axis. Note that even though the method signature is from matplotlib, after this operation ax is a
    # cartopy.mpl.geoaxes.GeoAxesSubplot object! This is a subclass of a matplotlib Axes class but not directly
    # compatible with one, so it means that this axis cannot, for example, be plotted using mplleaflet.
    ax = plt.subplot(111, projection=projection)

    # Set optional parameters.
    if extent:
        ax.set_extent(extent)
    if stock_image:
        ax.stock_img()
    if coastlines:
        ax.coastlines()

    # TODO: Refactor and include improvements from choropleth.

    # Draw. Notice that this scatter method's signature is attached to the axis instead of to the overall plot. This
    # is again because the axis is a special cartopy object.
    xs = np.array([p.x for p in df.geometry])
    ys = np.array([p.y for p in df.geometry])
    ax.scatter(xs, ys, transform=ccrs.PlateCarree(), **kwargs)
    plt.show()


def choropleth(df,
               data=None,
               scheme=None, k=5, cmap='Set1', vmin=None, vmax=None,
               spines=False,
               extent=None,
               stock_image=False, coastlines=False,
               projection=None,
               figsize=(8, 6),
               **kwargs):

    # Format the data to be displayed for input.
    if not data:
        nongeom = set(df.columns) - {df.geometry.name}
        if len(nongeom) > 1:
            raise ValueError("Ambiguous input: no 'data' parameter was specified and the inputted DataFrame has more "
                             "than one column of data.")
        else:
            data = df[list(nongeom)[0]]
    elif isinstance(data, str):
        data = df[data]

    # Initialize the figure.
    fig = plt.figure(figsize=figsize)

    # If we are not handed a projection we are in the PateCarree projection. In that case we can return a
    # `matplotlib` plot directly, which has the advantage of being native to e.g. mplleaflet.
    # TODO
    if not projection:
        pass

    projection = projection.load(df, {
        'central_longitude': lambda df: np.mean(np.array([p.x for p in df.geometry.centroid])),
        'central_latitude': lambda df: np.mean(np.array([p.y for p in df.geometry.centroid]))
    })

    # Set up the axis. Note that even though the method signature is from matplotlib, after this operation ax is a
    # cartopy.mpl.geoaxes.GeoAxesSubplot object! This is a subclass of a matplotlib Axes class but not directly
    # compatible with one, so it means that this axis cannot, for example, be plotted using mplleaflet.
    ax = plt.subplot(111, projection=projection)
    ax.margins(0.2)

    # Set extent.
    x_min_coord, x_max_coord, y_min_coord, y_max_coord = _get_envelopes_min_maxes(df.geometry.envelope.exterior)
    if extent:
        ax.set_extent(extent)
    else:
        ax.set_extent((x_min_coord, x_max_coord, y_min_coord, y_max_coord))

    # Set optional parameters.
    if stock_image:
        ax.stock_img()
    if coastlines:
        ax.coastlines()

    # Set up the colormap. This code is largely taken from geoplot's choropleth facilities, cf.
    # https://github.com/geopandas/geopandas/blob/master/geopandas/plotting.py#L253
    # If a scheme is provided we compute a distribution for the given data. If one is not provided we assume that the
    # input data is categorical.
    # TODO: The "scheme" specification used by geoplot is inconsistent, consider fixing that.
    if scheme:
        binning = __pysal_choro(data, scheme, k=k)
        values = binning.yb
        binedges = [binning.yb.min()] + binning.bins.tolist()
        categories = ['{0:.2f} - {1:.2f}'.format(binedges[i], binedges[i + 1])
                      for i in range(len(binedges) - 1)]
    else:
        values = data
        categories = None  # TODO: Implement.
    cmap = norm_cmap(values, cmap, Normalize, matplotlib.cm, vmin=vmin, vmax=vmax)

    # Set up spines. Cartopy by default generates and hides a plot's spines (cf.
    # https://github.com/SciTools/cartopy/blob/master/lib/cartopy/mpl/geoaxes.py#L972), we don't necessarily want that.
    # Instead what *is* enabled by default is a transparent background patch and an "outline" patch that forms a border.
    # This code removes the extraneous patches and optionally sets the axis.
    ax.background_patch.set_visible(False)
    ax.outline_patch.set_visible(False)
    if spines:
        ax.axes.get_xaxis().set_visible(True)
        ax.axes.get_yaxis().set_visible(True)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')
        # The default axis limits are equal to the extent of the plot, which is "some distribution" in the coordinate
        # reference system of the projection of the plot. For our purposes we'll just take them as being arbitrary and
        # overwrite the tick labels with our own computed values.
        x_min_proj, x_max_proj, y_min_proj, y_max_proj = ax.get_extent()
        x_transform = lambda x: ((x - x_min_proj) / (x_max_proj - x_min_proj)) * (x_max_coord - x_min_coord) + x_min_coord
        y_transform = lambda y: ((y - y_min_proj) / (y_max_proj - y_min_proj)) * (y_max_coord - y_min_coord) + y_min_coord
        ax.set_xticklabels(['{:.2f}'.format(x_transform(pos)) for pos in ax.get_xticks()])
        ax.set_yticklabels(['{:.2f}'.format(y_transform(pos)) for pos in ax.get_yticks()])

    # Finally we draw the features.
    for cat, geom in zip(values, df.geometry):
        features = ShapelyFeature([geom], ccrs.PlateCarree())
        ax.add_feature(features, facecolor=cmap.to_rgba(cat), **kwargs)
    plt.show()

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
