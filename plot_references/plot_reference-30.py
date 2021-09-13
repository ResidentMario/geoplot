import mapclassify as mc

scheme = mc.NaturalBreaks(injurious_collisions['NUMBER OF PERSONS INJURED'], k=3)
ax = gplt.voronoi(
    injurious_collisions.head(1000), projection=gcrs.AlbersEqualArea(),
    clip=boroughs.simplify(0.001),
    hue='NUMBER OF PERSONS INJURED', scheme=scheme, cmap='Reds',
    legend=True
)
gplt.polyplot(boroughs, ax=ax)