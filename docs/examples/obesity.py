# This examples was inspired by https://bl.ocks.org/mbostock/4055908

import pandas as pd
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

# load the data
obesity_by_state = pd.read_csv(gplt.datasets.get_path('obesity_by_state'), sep='\t')
contiguous_usa = gpd.read_file(gplt.datasets.get_path('contiguous_usa'))
contiguous_usa['Obesity Rate'] = contiguous_usa['State'].map(
    lambda state: obesity_by_state.query("State == @state").iloc[0]['Percent']
)


ax = gplt.cartogram(
    contiguous_usa,
    scale='Obesity Rate', limits=(0.75, 1),
    projection=gcrs.AlbersEqualArea(central_longitude=-98, central_latitude=39.5),
    hue='Obesity Rate', cmap='Reds', k=5,
    linewidth=0.5, trace_kwargs={'linewidth': 0.5},
    legend=True, legend_kwargs={'loc': 'lower right'}, legend_var='hue',
    figsize=(12, 12)
)

plt.title("Adult Obesity Rate by State, 2013")
plt.savefig("obesity.png", bbox_inches='tight', pad_inches=0.1)
