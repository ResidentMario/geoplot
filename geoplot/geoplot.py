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
    fig = plt.figure(figsize=figsize)
    # import pdb; pdb.set_trace()
    if not projection:
        projection = ccrs.PlateCarree()
    ax = plt.subplot(111, projection=projection)
    if extent:
        ax.set_extent(extent)
    else:
        # import pdb; pdb.set_trace()
        min_x = np.min([p.x for p in df.geometry])
        min_y = np.min([p.y for p in df.geometry])
        max_x = np.max([p.x for p in df.geometry])
        max_y = np.max([p.y for p in df.geometry])
        ax.set_extent((min_x, max_x, min_y, max_y))
    if stock_image:
        ax.stock_img()
    if coastlines:
        ax.coastlines()
    ax.scatter([p.x for p in df.geometry], [p.y for p in df.geometry], transform=ccrs.PlateCarree(), **kwargs)
    plt.show()