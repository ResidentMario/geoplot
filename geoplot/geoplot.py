import geopandas as gpd
import pandas as pd
from shapely import geometry
import matplotlib.pyplot as plt
import numpy as np
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

    # Handle the projection.
    # If we are not handed a projection we are in the PateCarree projection. In that case we can return a
    # `matplotlib` plot directly, which has the advantage of being native to e.g. mplleaflet.
    if not projection:
        return plt.scatter(xs, ys)

    # Otherwise, we have to deal with projection settings.
    proj_params = dict(projection.proj4_params)
    globe = projection.globe

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
    # Since we can't inject sensible defaults by default, we'll have to quietly reinitialize a different copy of the
    # object with our default central_latitude and/or central_longitude as inputs when needed. This means that, yes,
    # we effectively have to initialize our CRS twice!
    #
    # There's a lot unresolved awkardness here as well. I can account for and deal with the latitude and longitude
    # parameters, but I haven'y yet dealt with all of the *other* optional parameters, all of which would need to
    # also be detected off of `proj_params` and reinitialized themselves. I'm not even totally certain that I can
    # reverse engineer all of them!
    #
    # For now this is all a massive "to-do". I'm not sure what the most appealing API for this would be,
    # given the limitations present.
    import pdb; pdb.set_trace()
    if ('lon_0' in proj_params.keys() and proj_params['lon_0'] == 0) or\
       ('lat_0' in proj_params.keys() and proj_params['lat_0'] == 0):
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
        print({**proj_params, **projkwargs})
        projection = projection.__class__(**projkwargs, globe=globe)

    # Set up the axis...
    ax = plt.subplot(111, projection=projection)

    # Set optional params.
    if extent:
        ax.set_extent(extent)
    if stock_image:
        ax.stock_img()
    if coastlines:
        ax.coastlines()

    # Draw.
    ax.scatter(xs, ys, transform=ccrs.PlateCarree(), **kwargs)
    plt.show()