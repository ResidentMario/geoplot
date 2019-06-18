import geopandas as gpd
import geoplot as gplt
import matplotlib.pyplot as plt
import mplleaflet
from matplotlib.colors import LinearSegmentedColormap

# load the data
napoleon_troop_movements = gpd.read_file(gplt.datasets.get_path('napoleon_troop_movements'))
napoleon_troop_movements['from'] = napoleon_troop_movements.geometry.map(lambda v: v[0])
napoleon_troop_movements['to'] = napoleon_troop_movements.geometry.map(lambda v: v[1])

# plot the data with a custom colormap
colors = [(215/255, 193/255, 126/255), (37/255, 37/255, 37/255)]
cm = LinearSegmentedColormap.from_list('minard', colors)

gplt.sankey(
    napoleon_troop_movements,
    start='from', end='to',
    scale='survivors', limits=(0.5, 45),
    hue='direction',
    cmap=cm
)
fig = plt.gcf()
mplleaflet.save_html(fig, fileobj='minard-napoleon-russia.html')
