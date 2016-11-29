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
