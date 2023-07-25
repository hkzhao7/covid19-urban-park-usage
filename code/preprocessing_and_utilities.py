# data processing packages
import numpy as np
import pandas as pd

# stat packages
import scipy
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.stats import *

# viz packages
from matplotlib import pyplot as plt
from matplotlib_scalebar.scalebar import ScaleBar
import seaborn as sns

# geospatial packages
import cartopy.crs as ccrs
import cartopy
from cartopy.io.img_tiles import *
import geopy.distance
import geopandas as gpd


def process_cbg_connections(cbg_connections, nyc_boro_div):

    # add weighted travel distance (weight by the number of visitors)
    cbg_connections['distance_wtd'] = cbg_connections['distance_geod'] * cbg_connections['visitor_count']

    # get the county FIPS number
    # note: 
    cbg_connections['visitor_home_county'] = cbg_connections['GEOID_origin'].astype(str).str[:5]
    cbg_connections['visitor_dest_county'] = cbg_connections['GEOID_dest'].astype(str).str[:5]

    # convert the county FIPS number to the county name
    county_names = {'36005':'Bronx', '36047':'Brooklyn', '36061':'Manhattan', '36081':'Queens', '36085':'Staten Island'}
    cbg_connections['visitor_dest_county_name'] = cbg_connections['visitor_dest_county']
    cbg_connections.replace({'visitor_dest_county_name': county_names}, inplace=True)

    # drop records with no landuse (park type) information
    cbg_connections.rename(columns={'landuse': 'park_type'}, inplace=True)
    cbg_connections.dropna(subset=['park_type'], inplace=True)

    # clean nyc_boro_div
    nyc_boro_div.drop_duplicates(subset=['safegraph_place_id', 'parknum'], inplace=True)
    nyc_boro_div = nyc_boro_div[['safegraph_place_id', 'borough']]

    cbg_connections = pd.merge(cbg_connections, nyc_boro_div, on='safegraph_place_id')

    # add a borough column to cbg_connections
    cbg_connections['borough'].fillna(cbg_connections['visitor_dest_county_name'], inplace=True)
    
    # classify south of the 86th St as lower Manhattan
    replace_rules = {'Lower to Middle Manhattan': 'Lower Manhattan', 'Manhattan': 'Upper Manhattan'}
    cbg_connections.replace({'borough': replace_rules}, inplace=True)

    # change data type for county FIPS numbers
    cbg_connections['visitor_home_cbg'] = cbg_connections['visitor_home_cbg'].astype(str)
    cbg_connections['GEOID_origin'] = cbg_connections['GEOID_origin'].astype(str)
    cbg_connections['GEOID_dest'] = cbg_connections['GEOID_dest'].astype(str)
    

    return cbg_connections


def park_visits_fraction(cbg_connections_data, time_unit, id_cols, v_col, year=None):

    # calculate the visits to individual parks in each month
    park_visits = cbg_connections_data.drop_duplicates(subset=id_cols)
    park_visits = park_visits[['safegraph_place_id', 'park_name', 'park_type', 'date', 'year', 'month', 'raw_visit_counts', 'raw_visitor_counts', 'visitor_dest_county', 'visitor_dest_county_name', 'borough', 'poi_cbg', 'visitor_home_cbg', 'visitor_count']]
    park_visits = park_visits[park_visits['visitor_dest_county'].isin(['36005','36047','36061','36081','36085'])]
    park_visits = park_visits.groupby(['park_name', 'park_type', 'borough', 'date', 'year', 'month'], as_index=False)[v_col].sum()


    if time_unit == 'month':
        
        # calculate total monthly vistis to each type of park, and the total monthly park visits
        park_visits_total = park_visits.groupby(['date', 'year', 'month', 'park_type'], as_index=False)[v_col].sum()
        park_visits_monthly_total = park_visits_total.groupby(['date'], as_index=False)[v_col].sum()

        # calculate the fraction to the monthly total park visits
        park_visits_total = pd.merge(park_visits_total, park_visits_monthly_total, on=['date'], suffixes=['', '_monthly_total'])
        park_visits_total['fraction_to_monthly_total'] = park_visits_total[v_col] / park_visits_total[v_col + '_monthly_total'] * 100

        return park_visits, park_visits_total


    elif time_unit == 'year':
        
        # calculate total annual vistis to each type of park, and the total annual park visits
        park_visits_total = park_visits.groupby(['year', 'park_type'])[v_col].sum().reset_index()
        park_visits_annual_total = park_visits_total.groupby(['year'])[v_col].sum().reset_index()

        # calculate the fraction to the annual total park visits
        park_visits_total = pd.merge(park_visits_total, park_visits_annual_total, on=['year'], suffixes=['', '_annual_total'])
        park_visits_total['fraction_to_annual_total'] = park_visits_total[v_col] / park_visits_total[v_col + '_annual_total'] * 100

        # calculate the fraction for a specific year
        park_visits_annual_fraction = park_visits_total[park_visits_total['year'] == year].sort_values(by=['fraction_to_annual_total'], ascending=False).reset_index(drop=True)
        park_visits_annual_fraction['fraction_cumsum'] = park_visits_annual_fraction['fraction_to_annual_total'].cumsum()

        return park_visits_total, park_visits_annual_fraction



def remove_extreme_values(data, value_col):
    
    """
    get the middle 95% of data for each park_type, year and month,
    then concatenate the data chunks back together
    """

    data_list = []
    
    for park_type in data['park_type'].unique():
        # for year in data['year'].unique():
            for month in data['month'].unique():
                
                data_sub = data[(data['park_type'] == park_type)
                                # & (data['year'] == year)
                                & (data['month'] == month)]

                # select data for the middle 95%
                data_sub = data_sub[(data_sub[value_col] >= data_sub[value_col].quantile(0.025))
                                    & (data_sub[value_col] <= data_sub[value_col].quantile(0.975))]
                
                data_list.append(data_sub)
    
    return pd.concat(data_list, ignore_index=True)




def map_park_by_type(data, x_col, y_col, park_types, title):

    fig, axes = plt.subplots(figsize=(16,9), ncols=4, nrows=2, subplot_kw={'projection': ccrs.Mercator()})
    axes = axes.flatten()

    # get nyc borough boundaries data
    nyc_boro_bound = gpd.read_file(gpd.datasets.get_path('nybb'))
    nyc_boro_bound_wm = nyc_boro_bound.to_crs(ccrs.Mercator().proj4_init)    # convert to web mercator projection

    # construct a palette
    cmap = plt.cm.get_cmap('Set1')
    palette = {1:cmap.colors[0], -1:cmap.colors[1]}

    # calculate the extent of the map
    x0 = data[x_col].min()
    x1 = data[x_col].max()
    y0 = data[y_col].min()
    y1 = data[y_col].max()
    x_range = x1 - x0
    y_range = y1 - y0
    x0 = x0 - x_range * 0.2
    x1 = x1 + x_range * 0.1
    y0 = y0 - y_range * 0.1
    y1 = y1 + y_range * 0.2

    # prepare park annual visits data for easirer query
    # park_visits = park_annual_visits_data.set_index(['year','park_type'])

    for i, park_type in enumerate(park_types):

        data_sub = data[data['park_type'] == park_type]

        # calculate stats for labelling
        n_park = data_sub.shape[0]
        park_area_median = data_sub['area'].median()

        # construct the annotation string
        textstr = '\n'.join(('number of parks: {:.0f}'.format(n_park),
                             'median park area:',
                             '{:,.0f} $m^2$'.format(park_area_median)
                           ))

        # add basemap
        imagery = Stamen('toner-lite')
        axes[i].add_image(imagery, 11)

        # add nyc borough boundaries layer
        axes[i].add_geometries(nyc_boro_bound_wm['geometry'], alpha=0.5, edgecolor='k',
                               crs=ccrs.Mercator())

        # plot the visits change for individual parks
        sns.scatterplot(x=x_col, y=y_col, color='tab:red', size=6, alpha=0.7, zorder=10,
                        data=data_sub, ax=axes[i], transform=cartopy.crs.PlateCarree(), legend=None)


        # axes settings
        # sub_figure_label = '(' + chr(97+i) + ') ' + park_type
        axes[i].set_extent([x0, x1, y0, y1])
        axes[i].set_title(park_type, fontsize=15)
        # axes[i].legend(loc='upper left', frameon=True, fancybox=True)

        # place a text box in upper left in axes coords
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)    # matplotlib.patch.Patch properties
        axes[i].text(0.05, 0.95, textstr, transform=axes[i].transAxes, fontsize=14,
                     verticalalignment='top', bbox=props)


    # add a scale bar to the last subplot
    bbox = axes[-1].get_window_extent().transformed(axes[-1].transData.inverted())
    width, height = bbox.width, bbox.height

    dx = geopy.distance.geodesic((y0, x0), (y0, x1)).meters    # calculate the unit length at the latitude of the bottom of the plot
    dx /= width
    axes[-1].add_artist(ScaleBar(dx=dx, units="m", length_fraction=0.3, location='lower right', font_properties={'size':12}))


    plt.suptitle(title, fontsize=17)
    plt.tight_layout();