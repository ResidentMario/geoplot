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

import cartopy.crs as ccrs

# TODO: RotatedPole

class PlateCarree:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class LambertCylindrical:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class Mercator:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class Miller:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class Mollweide:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class Robinson:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class Sinusoidal:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class InterruptedGoodeHomolosine:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class Geostationary:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class NorthPolarStereo:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class SouthPolarStereo:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_longitude': centerings['central_longitude']})


class Gnomonic:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings):
        return _generic_load(self, df, {'central_latitude': centerings['central_latitude']})


class AlbersEqualArea:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)


class AzimuthalEquidistant:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)


class LambertConformal:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)


class Orthographic:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)


class Stereographic:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)


class TransverseMercator:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)


class LambertAzimuthalEqualArea:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, centerings): return _generic_load(self, df, centerings)


class UTM:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, _): return _generic_load(self, df, dict())


class OSGB:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, _): return _generic_load(self, df, dict())


class EuroPP:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, _): return _generic_load(self, df, dict())


class OSNI:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df, _): return _generic_load(self, df, dict())


def _generic_load(proj, df, centerings):
    centering_variables = dict()
    for key, func in centerings.items():
        centering_variables[key] = func(df)
    return getattr(ccrs, proj.__class__.__name__)(**{**centering_variables, **proj.args})
