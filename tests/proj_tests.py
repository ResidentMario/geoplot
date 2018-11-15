"""
Test that projections in `geoplot` function correctly.
"""

import pytest
import geoplot as gplt
import geoplot.crs as gcrs
import geopandas as gpd
import matplotlib.pyplot as plt


@pytest.fixture(scope="module")
def countries():
    return gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("proj", [
    # TODO: Fix errors in the projections that do not currently work.
    gcrs.PlateCarree(),
    # gcrs.LambertCylindrical(),
    gcrs.Mercator(),
    gcrs.Miller(),
    gcrs.Mollweide(),
    gcrs.Robinson(),
    gcrs.Sinusoidal(),
    gcrs.InterruptedGoodeHomolosine(),
    gcrs.Geostationary(),
    gcrs.NorthPolarStereo(),
    gcrs.SouthPolarStereo(),
    gcrs.Gnomonic(),
    gcrs.AlbersEqualArea(),
    gcrs.AzimuthalEquidistant(),
    gcrs.LambertConformal(),
    gcrs.Orthographic(),
    gcrs.Stereographic(),
    gcrs.TransverseMercator(),
    gcrs.LambertAzimuthalEqualArea()
    # # TODO: Include other new ones.
])
def test_basic_global_projections(proj, countries):
    gplt.polyplot(countries, proj)
    ax = plt.gca()
    ax.set_global()

    return plt.gcf()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("proj", [
    gcrs.EuroPP(),
    gcrs.OSGB(),
])
def test_basic_non_global_projections(proj, countries):
    gplt.polyplot(gpd.GeoDataFrame(geometry=[]), proj)
    # TODO: gplt.polyplot(countries, proj)
    return plt.gcf()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("proj", [
    gcrs.PlateCarree(central_longitude=45),
    # gcrs.LambertCylindrical(central_longitude=45),
    gcrs.Mercator(central_longitude=45),
    gcrs.Miller(central_longitude=45),
    gcrs.Mollweide(central_longitude=45),
    gcrs.Robinson(central_longitude=45),
    gcrs.Sinusoidal(central_longitude=45),
    gcrs.InterruptedGoodeHomolosine(central_longitude=45),
    gcrs.Geostationary(central_longitude=45),
    gcrs.NorthPolarStereo(central_longitude=45),
    gcrs.SouthPolarStereo(central_longitude=45),
    gcrs.Gnomonic(central_latitude=45),
    gcrs.AlbersEqualArea(central_longitude=45, central_latitude=45),
    gcrs.AzimuthalEquidistant(central_longitude=45, central_latitude=45),
    gcrs.LambertConformal(central_longitude=45, central_latitude=45),
    gcrs.Orthographic(central_longitude=45, central_latitude=45),
    gcrs.Stereographic(central_longitude=45, central_latitude=45),
    gcrs.TransverseMercator(central_longitude=45, central_latitude=45),
    gcrs.LambertAzimuthalEqualArea(central_longitude=45, central_latitude=45)
])
def test_fully_parameterized_global_projections(proj, countries):
    gplt.polyplot(countries, proj)
    ax = plt.gca()
    ax.set_global()

    return plt.gcf()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("proj", [
    gcrs.AlbersEqualArea(central_longitude=45),
    gcrs.AlbersEqualArea(central_latitude=45),
    gcrs.AzimuthalEquidistant(central_longitude=45),
    gcrs.AzimuthalEquidistant(central_latitude=45),
    gcrs.LambertConformal(central_longitude=45),
    gcrs.LambertConformal(central_latitude=45),
    gcrs.Orthographic(central_longitude=45),
    gcrs.Orthographic(central_latitude=45),
    gcrs.Stereographic(central_longitude=45),
    gcrs.Stereographic(central_latitude=45),
    gcrs.TransverseMercator(central_longitude=45),
    gcrs.TransverseMercator(central_latitude=45),
    gcrs.LambertAzimuthalEqualArea(central_longitude=45),
    gcrs.LambertAzimuthalEqualArea(central_latitude=45)
])
def test_partially_parameterized_global_projections(proj, countries):
    gplt.polyplot(countries, proj)
    ax = plt.gca()
    ax.set_global()

    return plt.gcf()


@pytest.mark.mpl_image_compare
@pytest.mark.parametrize("proj", [
    gcrs.PlateCarree(),
    gcrs.LambertCylindrical(),
    gcrs.Mercator(),
    gcrs.Miller(),
    gcrs.Mollweide(),
    gcrs.Robinson(),
    gcrs.Sinusoidal(),
    pytest.param(gcrs.InterruptedGoodeHomolosine(), marks=pytest.mark.xfail),
    pytest.param(gcrs.Geostationary(), marks=pytest.mark.xfail),
    gcrs.NorthPolarStereo(),
    gcrs.SouthPolarStereo(),
    pytest.param(gcrs.Gnomonic(), marks=pytest.mark.xfail),
    gcrs.AlbersEqualArea(),
    gcrs.AzimuthalEquidistant(),
    gcrs.LambertConformal(),
    pytest.param(gcrs.Orthographic(), marks=pytest.mark.xfail),
    gcrs.Stereographic(),
    pytest.param(gcrs.TransverseMercator(), marks=pytest.mark.xfail),
    gcrs.LambertAzimuthalEqualArea()
    # # TODO: Include other new ones.
])
def test_subplots_global_projections(proj, countries):
    gplt.polyplot(countries, proj, ax=plt.subplot(2, 1, 1, projection=proj)).set_global()
    gplt.polyplot(countries, proj, ax=plt.subplot(2, 1, 2, projection=proj)).set_global()

    return plt.gcf()
