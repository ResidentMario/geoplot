import cartopy.crs as ccrs
import numpy as np


# TODO: RotatedPole

class PlateCarree:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class LambertCylindrical:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class Mercator:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class Miller:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class Mollweide:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class Robinson:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class Sinusoidal:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class InterruptedGoodeHomolosine:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class Geostationary:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class NorthPolarStereo:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class SouthPolarStereo:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True)


class Gnomonic:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_latitude=True)


class AlbersEqualArea:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True, has_central_latitude=True)


class AzimuthalEquidistant:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True, has_central_latitude=True)


class LambertConformal:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True, has_central_latitude=True)


class Orthographic:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True, has_central_latitude=True)


class Stereographic:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True, has_central_latitude=True)


class TransverseMercator:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True, has_central_latitude=True)


class LambertAzimuthalEqualArea:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df, has_central_longitude=True, has_central_latitude=True)


class UTM:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df)


class OSGB:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df)


class EuroPP:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df)


class OSNI:
    def __init__(self, **kwargs): self.args = kwargs

    def load(self, df): return _generic_load(self, df)

def _generic_load(proj, df,
                  has_central_longitude=False, has_central_latitude=False):
    centering_variables = dict()
    if has_central_longitude:
        if "central_longitude" not in proj.args.keys():
            central_longitude = np.mean(np.array([p.x for p in df.geometry]))
        else:
            central_longitude = proj.args['central_longitude']
            proj.args.pop('central_longitude')
        centering_variables['central_longitude'] = central_longitude
    if has_central_latitude:
        if "central_latitude" not in proj.args.keys():
            central_latitude = np.mean(np.array([p.x for p in df.geometry]))
        else:
            central_latitude = proj.args['central_latitude']
            proj.args.pop('central_latitude')
        centering_variables['central_latitude'] = central_latitude
    import pdb; pdb.set_trace()
    return getattr(ccrs, proj.__class__.__name__)(**{**centering_variables, **proj.args})
