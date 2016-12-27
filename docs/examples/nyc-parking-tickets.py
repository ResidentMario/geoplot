import sys; sys.path.insert(0, '../')
import pandas as pd
import geopandas as gpd
import geoplot as gplt
import geoplot.crs as gcrs
import matplotlib.pyplot as plt
import numpy as np


# This example inspired by:
# http://iquantny.tumblr.com/post/84393789169/californians-love-brooklyn-new-jerseyans-love
# The dataset provided is an aggregation based on:
# https://data.cityofnewyork.us/-/Parking-Violations-Issued-Fiscal-Year-2016


# Load the data.
precincts = gpd.read_file("../../data/nyc_precincts/Police Precincts.geojson")
precincts['precinct'] = precincts['precinct'].astype(int)
tickets = pd.read_csv("../../data/nyc_parking/Aggregated_Parking_Violations_-_Fiscal_Year_2016.csv", index_col=0)
# Due to an error on my part, the result is not quite grouped correctly. The following code bit fixes the issue.
tickets = tickets.groupby(['Precinct', 'State Name'])[['Count']].sum().reset_index()
tickets = tickets[tickets['Precinct'].isin(precincts['precinct'].values)]
tickets['Precinct'] = tickets['Precinct'].astype(int)
boroughs = gpd.read_file("../../data/nyc_boroughs/boroughs.geojson")

state_names = np.unique(tickets['State Name'].values)
precinct_ticket_totals = tickets.groupby('Precinct').sum()
state_ticket_totals = tickets.groupby('State Name').sum()
precincts = precincts.set_index('precinct')


# The following function automatically munges the data as appropriate for a particular state.
def tickets_by_precinct(state):
    """
    Returns data with the percentage of tickets issued in the given census tract out of all tickets issued to that
    *precinct*
    """
    state_tickets = tickets[tickets['State Name'] == state].set_index('Precinct')

    def get_precinct_ticket_percentage(srs):
        precinct = srs.name
        state_count = srs['Count']
        precinct_count = precinct_ticket_totals.loc[precinct]
        return state_count / precinct_count

    precinct_ticket_percentages = state_tickets.apply(get_precinct_ticket_percentage, axis='columns')

    def get_geom(precinct_num):
        return precincts.loc[precinct_num].geometry

    geom = precinct_ticket_percentages.index.map(get_geom)
    geo_data = gpd.GeoDataFrame(data=precinct_ticket_percentages, geometry=geom)
    return geo_data


# This function defines a method for plotting the data.
def plot_state_to_ax(state, ax):
    n = state_ticket_totals.loc[state]['Count']
    gplt.choropleth(tickets_by_precinct(state), projection=gcrs.AlbersEqualArea(), cmap='Blues',
                    linewidth=0.0, ax=ax)
    gplt.polyplot(boroughs, projection=gcrs.AlbersEqualArea(), edgecolor='black', linewidth=0.5, ax=ax)
    ax.set_title("{0} (n={1})".format(state, n))


# Finally, plot the data.
f, axarr = plt.subplots(2, 2, figsize=(12, 12), subplot_kw={
    'projection': gcrs.AlbersEqualArea(central_latitude=40.7128, central_longitude=-74.0059)
})


plt.suptitle('Parking Tickets Issued to State by Precinct, 2016', fontsize=16)
plt.subplots_adjust(top=0.95)
plot_state_to_ax('New York', axarr[0][0])
plot_state_to_ax('New Jersey', axarr[0][1])
plot_state_to_ax('Pennsylvania', axarr[1][0])
plot_state_to_ax('Connecticut', axarr[1][1])

plt.savefig("nyc-parking-tickets.png", bbox_inches='tight')
