ax = gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea(), zorder=1)
gplt.kdeplot(collisions, cmap='Reds', shade=True, clip=boroughs, ax=ax)