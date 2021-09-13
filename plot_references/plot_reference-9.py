collisions = gpd.read_file(gplt.datasets.get_path('nyc_collision_factors'))

ax = gplt.webmap(boroughs, projection=gcrs.WebMercator())
gplt.pointplot(
    collisions[collisions['BOROUGH'].notnull()],
    hue='BOROUGH', ax=ax, legend=True
)