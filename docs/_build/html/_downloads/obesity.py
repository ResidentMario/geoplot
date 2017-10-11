# This examples was inspired by https://bl.ocks.org/mbostock/4055908

# Load the data.
import geopandas as gpd
from quilt.data.ResidentMario import geoplot_data

obesity = geoplot_data.obesity_by_state()
contiguous_usa = gpd.read_file(geoplot_data.contiguous_usa())
contiguous_usa['State'] = [
    'Minnesota', 'Montana', 'North Dakota', 'Idaho', 'Washington', 'Arizona',
    'California', 'Colorado', 'Nevada', 'New Mexico', 'Oregon', 'Utah', 'Wyoming',
    'Arkansas', 'Iowa', 'Kansas', 'Missouri', 'Nebraska', 'Oklahoma', 'South Dakota',
    'Louisiana', 'Texas', 'Connecticut', 'Massachusetts', 'New Hampshire',
    'Rhode Island', 'Vermont', 'Alabama', 'Florida', 'Georgia', 'Mississippi',
    'South Carolina', 'Illinois', 'Indiana', 'Kentucky', 'North Carolina', 'Ohio',
    'Tennessee', 'Virginia', 'Wisconsin', 'West Virginia', 'Delaware', 'District of Columbia',
    'Maryland', 'New Jersey', 'New York', 'Pennsylvania', 'Maine', 'Michigan'
]
contiguous_usa['Obesity Rate'] = contiguous_usa['State'].map(
    lambda state: obesity.query("State == @state").iloc[0]['Percent']
)


# Plot the data.
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt

ax = gplt.cartogram(contiguous_usa, scale='Obesity Rate',
                    projection=gcrs.AlbersEqualArea(central_longitude=-98, central_latitude=39.5),
                    limits=(0.75, 1),
                    linewidth=0.5, facecolor='steelblue', trace_kwargs={'linewidth': 0.5})
ax.set_ylim((-1597757.3894385984, 1457718.4893930717))
plt.title("Adult Obesity Rate by State, 2013")
plt.savefig("obesity.png", bbox_inches='tight', pad_inches=0.1)