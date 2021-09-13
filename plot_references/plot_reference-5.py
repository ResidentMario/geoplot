gplt.pointplot(
    cities, projection=gcrs.AlbersEqualArea(),
    hue='ELEV_IN_FT', scale='ELEV_IN_FT', limits=(1, 10), cmap='inferno_r',
    legend=True, legend_var='scale'
)