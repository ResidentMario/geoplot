ax = gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea(), zorder=1)
gplt.kdeplot(collisions, cmap='Reds', shade=True, thresh=0.05,
             clip=boroughs, ax=ax)