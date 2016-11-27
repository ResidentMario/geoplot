import cartopy.crs as ccrs


class PlateCarree(ccrs.PlateCarree):
    def __init__(self, central_longitude=None, globe=None):
        if not central_longitude:
            pass
