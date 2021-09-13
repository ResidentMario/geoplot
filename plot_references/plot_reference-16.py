ax = gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea())
gplt.kdeplot(collisions, n_levels=20, cmap='Reds', ax=ax)