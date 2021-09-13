import mapclassify as mc

scheme = mc.FisherJenks(contiguous_usa['population'], k=5)
gplt.choropleth(
    contiguous_usa, hue='population', projection=gcrs.AlbersEqualArea(),
    edgecolor='white', linewidth=1,
    cmap='Greens',
    legend=True, legend_kwargs={'loc': 'lower left'},
    scheme=scheme
)