import mapclassify as mc

scheme = mc.Quantiles(cities['ELEV_IN_FT'], k=5)
gplt.pointplot(
    cities, projection=gcrs.AlbersEqualArea(),
    hue='ELEV_IN_FT', scheme=scheme, cmap='inferno_r',
    legend=True
)