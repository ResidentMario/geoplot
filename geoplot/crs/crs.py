"""
This module defines the ``geoplot`` coordinate reference system classes, wrappers on ``cartopy.crs`` objects meant
to be used as parameters to the ``projection`` parameter of all front-end ``geoplot`` outputs.

This was necessary because ``cartopy.crs`` objects do not allow modifications in place. cf.
http://stackoverflow.com/questions/40822241/seemingly-immutable-dict-in-object-instance/40822473

For the list of Cartopy CRS objects this module derives from, refer to
http://scitools.org.uk/cartopy/docs/latest/crs/projections.html
"""

import cartopy.crs as ccrs
import geopandas as gpd

# TODO: RotatedPole


class PlateCarree:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class LambertCylindrical:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class Mercator:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class Miller:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class Mollweide:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class Robinson:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class Sinusoidal:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class InterruptedGoodeHomolosine:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class Geostationary:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class NorthPolarStereo:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class SouthPolarStereo:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class Gnomonic:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_latitude': centerings['central_latitude']})

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class AlbersEqualArea:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class AzimuthalEquidistant:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class LambertConformal:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class Orthographic:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class Stereographic:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class TransverseMercator:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class LambertAzimuthalEqualArea:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class UTM:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, _): return _generic_load(self, df, dict())

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class OSGB:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, _): return _generic_load(self, df, dict())

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class EuroPP:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, _): return _generic_load(self, df, dict())

    def _as_mpl_axes(self): return _as_mpl_axes(self)


class OSNI:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, _): return _generic_load(self, df, dict())

    def _as_mpl_axes(self): return _as_mpl_axes(self)


def _generic_load(proj, df, centerings):
    """
    A moderately mind-bendy meta-method which abstracts the internals of individual projections' load procedures.

    Parameters
    ----------
    proj : geoplot.crs object instance
        A disguised reference to ``self``.
    df : GeoDataFrame
        The GeoDataFrame which has been passed as input to the plotter at the top level. This data is needed to
        calculate reasonable centering variables in cases in which the user does not already provide them; which is,
        incidentally, the reason behind all of this funny twice-instantiation loading in the first place.
    centerings: dct
        A dictionary containing names and centering methods. Certain projections have certain centering parameters
        whilst others lack them. For example, the geospatial projection contains both ``central_longitude`` and
        ``central_latitude`` instance parameter, which together control the center of the plot, while the North Pole
        Stereo projection has only a ``central_longitude`` instance parameter, implying that latitude is fixed (as
        indeed it is, as this projection is centered on the North Pole!).

        A top-level centerings method is provided in each of the ``geoplot`` top-level plot functions; each of the
        projection wrapper classes defined here in turn selects the functions from this list relevent to this
        particular instance and passes them to the ``_generic_load`` method here.

        We then in turn execute these functions to get defaults for our ``df`` and pass them off to our output
        ``cartopy.crs`` instance.

    Returns
    -------
    crs : ``cartopy.crs`` object instance
        Returns a ``cartopy.crs`` object instance whose appropriate instance variables have been set to reasonable
        defaults wherever not already provided by the user.
    """
    centering_variables = dict()
    for key, func in centerings.items():
        centering_variables[key] = func(df)
    return getattr(ccrs, proj.__class__.__name__)(**{**centering_variables, **proj.args})


def _as_mpl_axes(proj):
    """
    Another moderately mind-bendy method. When ``matplotlib`` is provided a projection via a ``projection`` keyword
    argument, it expects to get something with a callable ``as_mpl_axes`` method. The precise details of what this
    method does, exactly, are not important: it suffices to know that every ``cartopy`` coordinate reference system
    object has one.

    When we pass a ``geoplot.crs`` crs object to a ``geoplot`` function, the loading and centering of the data
    occurs automatically (using the function defined immediately above). Since we control what ``geoplot`` does at
    execution, we gracefully integrate this two-step procedure into the function body.

    But there are also use cases outside of our control in which we are forced to pass a ``geoplot.crs`` object
    without having first called ``load``: most prominently, when creating a plot containing subplots, the "overall"
    projection must be pre-loaded. It's possible to get around this by using ``cartopy.crs`` objects instead,
    but this is inelegant. This method is a better way: when a ``cartopy.crs`` object called by ``matplotlib``,
    it silently swaps itself out for a vanilla version of its ``cartopy.crs`` mirror, and calls that function's
    ``_as_mpl_axes`` instead.

    Parameters
    ----------
    proj : geoplot.crs projection instance
        The instance in question (self, in the method body).

    Returns
    -------
    Mutates into a ``cartopy.crs`` object and returns the result of executing ``_as_mpl_axes`` on that object instead.

    """
    proj = proj.load(gpd.GeoDataFrame(), dict())
    return proj._as_mpl_axes()
