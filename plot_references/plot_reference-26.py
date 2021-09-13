import mapclassify as mc

scheme = mc.Quantiles(la_flights['Passengers'], k=5)
ax = gplt.sankey(
    la_flights, projection=gcrs.Mollweide(),
    scale='Passengers', limits=(1, 10),
    hue='Passengers', cmap='Greens', scheme=scheme, legend=True
)
gplt.polyplot(
    world, ax=ax, facecolor='lightgray', edgecolor='white'
)
ax.set_global(); ax.outline_patch.set_visible(True)