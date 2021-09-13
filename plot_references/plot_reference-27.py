import mapclassify as mc

scheme = mc.Quantiles(la_flights['Passengers'], k=5)
ax = gplt.sankey(
    la_flights, projection=gcrs.Mollweide(),
    scale='Passengers', limits=(1, 10),
    hue='Passengers', scheme=scheme, cmap='Greens',
    legend=True, legend_kwargs={'loc': 'lower left'}
)
gplt.polyplot(
    world, ax=ax, facecolor='lightgray', edgecolor='white'
)
ax.set_global(); ax.outline_patch.set_visible(True)