import sys; sys.path.insert(0, '../../')
import geoplot as gplt
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point
import pandas as pd
import mplleaflet

# Note that in order to get the webmap working inside of ReStructed Text, the format in which this guide is written,
# the output had to be hand-modified slightly. This trick:
# http://stackoverflow.com/questions/41272106/directive-for-including-a-raw-html-file-in-a-restructured-text-page
# ...was employed, and the output HTML was further modified slightly so that it could function within this page (
# removing the <body> tag etc.).

# Shape the data.
troop_positions = pd.read_fwf("../../data/napoloen/troops.txt")
troop_positions = gpd.GeoDataFrame(data=troop_positions,
                                   geometry=troop_positions\
                                       .apply(lambda srs: Point(srs['long'], srs['lat']),
                                              axis='columns'))

subsrs = []
for a, b in zip(range(len(troop_positions) - 1), range(1, len(troop_positions))):
    srs = troop_positions.iloc[b]
    srs = srs.rename({'geometry': 'from'})
    srs['to'] = troop_positions.iloc[a].geometry
    subsrs.append(srs)
troop_movements = pd.concat(subsrs, axis=1).T
troop_movements = troop_movements[['survivors', 'direction', 'group', 'from', 'to']]
troop_movements['direction'] = troop_movements.direction.map(lambda d: 0 if d == 'A' else 1)


# Custom colormap, to match the one that Minard uses.
from matplotlib.colors import LinearSegmentedColormap
colors = [(215/255, 193/255, 126/255), (37/255, 37/255, 37/255)]
cm = LinearSegmentedColormap.from_list('minard', colors)


# Plot the data.
gplt.sankey(troop_movements, start='from', end='to',
            scale='survivors', limits=(0.5, 45),
            hue='direction', categorical=True, cmap=cm)
fig = plt.gcf()
mplleaflet.save_html(fig, fileobj='minard-napoleon-russia.html')