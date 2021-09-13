gplt.quadtree(
    collisions, nmax=1,
    projection=gcrs.AlbersEqualArea(), clip=boroughs.simplify(0.001),
    hue='NUMBER OF PEDESTRIANS INJURED', cmap='Reds',
    edgecolor='white', legend=True,
)