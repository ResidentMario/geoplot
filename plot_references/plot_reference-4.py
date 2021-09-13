gplt.pointplot(
    cities, projection=gcrs.AlbersEqualArea(),
    hue='ELEV_IN_FT',
    legend=True, legend_kwargs={'orientation': 'horizontal'},
    edgecolor='lightgray', linewidth=0.5
)