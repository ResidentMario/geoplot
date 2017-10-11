# Load the data.
import geopandas as gpd
from quilt.data.ResidentMario import geoplot_data

troop_movements = gpd.read_file(geoplot_data.napoleon_troop_movements())
troop_movements['from'] = troop_movements.geometry.map(lambda v: v[0])
troop_movements['to'] = troop_movements.geometry.map(lambda v: v[1])


# Plot the data. We'll use a custom colormap, to match the one that Minard uses.
import geoplot as gplt
import matplotlib.pyplot as plt
import mplleaflet

from matplotlib.colors import LinearSegmentedColormap
colors = [(215/255, 193/255, 126/255), (37/255, 37/255, 37/255)]
cm = LinearSegmentedColormap.from_list('minard', colors)


gplt.sankey(troop_movements, start='from', end='to',
            scale='survivors', limits=(0.5, 45),
            hue='direction', categorical=True, cmap=cm)
fig = plt.gcf()
mplleaflet.save_html(fig, fileobj='minard-napoleon-russia.html')