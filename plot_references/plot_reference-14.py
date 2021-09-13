import mapclassify as mc

scheme = mc.FisherJenks(contiguous_usa['population'], k=5)
gplt.choropleth(
    contiguous_usa, hue='population', projection=gcrs.AlbersEqualArea(),
    edgecolor='white', linewidth=1,
    cmap='Greens', legend=True, legend_kwargs={'loc': 'lower left'},
    scheme=scheme,
    legend_labels=[
        '<3 million', '3-6.7 million', '6.7-12.8 million',
        '12.8-25 million', '25-37 million'
    ]
)