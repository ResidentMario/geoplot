ax = gplt.voronoi(
    injurious_collisions.head(100),
    clip=boroughs.simplify(0.001), projection=gcrs.AlbersEqualArea()
)
gplt.polyplot(boroughs, ax=ax)