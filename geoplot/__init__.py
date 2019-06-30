from .geoplot import (
    pointplot, polyplot, choropleth, cartogram, kdeplot, sankey, voronoi, quadtree, __version__
)
from .crs import (
    PlateCarree, LambertCylindrical, Mercator, Miller, Mollweide, Robinson, Sinusoidal,
    InterruptedGoodeHomolosine, Geostationary, NorthPolarStereo, SouthPolarStereo, Gnomonic,
    AlbersEqualArea, AzimuthalEquidistant, LambertConformal, Orthographic, Stereographic,
    TransverseMercator, LambertAzimuthalEqualArea, UTM, OSGB, EuroPP, OSNI
)
from .utils import (
    gaussian_points, gaussian_polygons, gaussian_multi_polygons, uniform_random_global_points,
    uniform_random_global_network, classify_clusters
)
from .datasets import get_path
