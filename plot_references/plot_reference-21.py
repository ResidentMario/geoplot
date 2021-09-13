import mapclassify as mc

scheme = mc.Quantiles(contiguous_usa['population'], k=5)
ax = gplt.cartogram(
    contiguous_usa, scale='population', projection=gcrs.AlbersEqualArea(),
    legend=True, legend_kwargs={'bbox_to_anchor': (1, 0.9)}, legend_var='hue',
    hue='population', scheme=scheme, cmap='Greens'
)
gplt.polyplot(contiguous_usa, facecolor='lightgray', edgecolor='white', ax=ax)