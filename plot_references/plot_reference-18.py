ax = gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea(), zorder=1)
gplt.kdeplot(collisions, cmap='Reds', shade=True, shade_lowest=True,
             clip=boroughs, ax=ax)