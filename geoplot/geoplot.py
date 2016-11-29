import geopandas as gpd
import pandas as pd
from shapely import geometry
import matplotlib.pyplot as plt
import numpy as np
from cartopy.feature import ShapelyFeature
import cartopy.crs as ccrs


def pointplot(df,
              extent=None,
              stock_image=False, coastlines=False,
              projection=None,
              figsize=(12, 4),
              **kwargs):
    # Initialize the figure.
    fig = plt.figure(figsize=figsize)

    # Initialize coordinate arrays. We do this only once, here, for efficiency.
    xs = np.array([p.x for p in df.geometry])
    ys = np.array([p.y for p in df.geometry])

    # TODO
    # # If we are not handed a projection we are in the PateCarree projection. In that case we can return a
    # # `matplotlib` plot directly, which has the advantage of being native to e.g. mplleaflet.
    # if not projection:
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
    projection = projection.load(df)

    # Set up the axis. Note that even though the method signature is from matplotlib, after this operation ax is a
    # cartopy.mpl.geoaxes.GeoAxesSubplot object! This is a subclass of a matplotlib Axes class but not directly
    # compatible with one, so it means that this axis cannot, for example, be plotted using mplleaflet.
    ax = plt.subplot(111, projection=projection)

    # Set optional params.
    if extent:
        ax.set_extent(extent)
    if stock_image:
        ax.stock_img()
    if coastlines:
        ax.coastlines()

    # Draw. Notice that this scatter method's signature is attached to the axis instead of to the overall plot. This
    # is again because the axis is a special cartopy object.
    ax.scatter(xs, ys, transform=ccrs.PlateCarree(), **kwargs)
    plt.show()


def choropleth(df,
               data=None,
               extent=None,
               stock_image=False, coastlines=False,
               projection=None,
               figsize=(12, 4),
               **kwargs):

    # Initialize the figure.
    fig = plt.figure(figsize=figsize)

    # If we are not handed a projection we are in the PateCarree projection. In that case we can return a
    # `matplotlib` plot directly, which has the advantage of being native to e.g. mplleaflet.
    if not projection:
        pass

    # Otherwise, assemble the defaults.
    proj_params = dict(projection.proj4_params)
    globe = projection.globe

    if ('lon_0' in proj_params.keys() and proj_params['lon_0'] == 0) or\
       ('lat_0' in proj_params.keys() and proj_params['lat_0'] == 0):
        xs = np.array([p.x for p in df.geometry.centroid])
        ys = np.array([p.y for p in df.geometry.centroid])
        projkwargs = dict()
        if 'lon_0' in proj_params.keys() and proj_params['lon_0'] == 0:
            projkwargs['central_longitude'] = np.mean(xs)
            proj_params.pop('lon_0')
        else:
            pass
        if 'lat_0' in proj_params.keys() and proj_params['lat_0'] == 0:
            projkwargs['central_latitude'] = np.mean(ys)
            proj_params.pop('lat_0')
        else:
            pass
        projection = projection.__class__(**projkwargs, globe=globe)

    # Initialize the cartopy axis.
    ax = plt.subplot(111, projection=projection)

    # Set optional params.
    if extent:
        ax.set_extent(extent)
    else:
        envelopes = df.geometry.envelope.exterior
        import pdb; pdb.set_trace()
        xmin = np.min(envelopes.map(lambda linearring: np.min([linearring.coords[1][0],
                                                               linearring.coords[2][0],
                                                               linearring.coords[3][0],
                                                               linearring.coords[4][0]])))
        ymin = np.min(envelopes.map(lambda linearring: np.min([linearring.coords[1][1],
                                                               linearring.coords[2][1],
                                                               linearring.coords[3][1],
                                                               linearring.coords[4][1]])))
        xmax = np.max(envelopes.map(lambda linearring: np.max([linearring.coords[1][0],
                                                               linearring.coords[2][0],
                                                               linearring.coords[3][0],
                                                               linearring.coords[4][0]])))
        ymax = np.max(envelopes.map(lambda linearring: np.max([linearring.coords[1][1],
                                                               linearring.coords[2][1],
                                                               linearring.coords[3][1],
                                                               linearring.coords[4][1]])))
        ax.set_extent((xmin, xmax, ymin, ymax))
    if stock_image:
        ax.stock_img()
    if coastlines:
        ax.coastlines()

    # Draw. To do this, we'll have to bastardize parts of the geopandas drawing API, taking code from here:
    # https://github.com/geopandas/geopandas/blob/46e50fe5a772944b325bc3c8806f4f96da76a0d8/geopandas/plotting.py#L120
    # TODO: Implement this.
    features = ShapelyFeature(df.geometry, ccrs.PlateCarree())
    ax.add_feature(features, **kwargs)
    plt.show()

    pass