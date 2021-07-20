# -*- coding: utf-8 -*-
"""
@author: HZU
"""
import pandas as pd
import numpy as np
from scipy import stats



postal_code = pd.read_csv('postal_code_belgium.csv', sep=',')

def outliers(df, strategy="Do not remove"):
    if strategy[:6] =="Remove":
        df = df[(np.abs(stats.zscore(df['actual_price'])) < int(strategy[-1]))]
        df = df[(np.abs(stats.zscore(df['area'])) < int(strategy[-1]))]
    return df

def fixing_lon_lat(df):
    #Finding the locations that need to adjust longitud and latitud
    list_postal_codes_to_fill = df.loc[df['location_lat'].isnull()]['location'].values.tolist()
    list_of_index_to_fix_lat_lon = df.loc[df['location_lat'].isnull()].index.values.astype(int)
    
    p = []
    for i in range(len(list_postal_codes_to_fill)):
        p = postal_code[postal_code.eq(list_postal_codes_to_fill[i]).any(1)].values.tolist()
        df.loc[list_of_index_to_fix_lat_lon[i], ['location_lon']] = p[0][2]
        df.loc[list_of_index_to_fix_lat_lon[i], ['location_lat']] = p[0][3]
        p = []

    return df

def cleaning_data(df):
    """
    Parameters
    ----------
    df : dataframe
        this is data base with all the data before to be cleaned.

    Returns
    -------
    df : dataframe
        We will drop all the values that are NA for the columns that we are interested.
        Also we will drop all the duplicates
        We will finish with the columns that will be used in the analysis.
    """
    df = df.dropna(subset=['actual_price'])
    df = df.dropna(subset=['area'])
    # df = df.dropna(subset=['building_condition'])
    df = df.dropna(subset=['location_lat']) 
    df = df.dropna(subset=['location_lon'])     
    df = df.drop_duplicates(subset=['prop_id'])
    df = df.fillna(0)
    df['price_x_m2'] = df['actual_price'] / df['area']
    df = df.drop(columns=['point_of_interest', 'subtype', 'old_price_value', 'room_number',
           'statistics_bookmark_count', 'statistics_view_count', 'creation_date', 
           'expiration_date', 'last_modification_date', 'kitchen_equipped', 'furnished', 'fireplace',
           'terrace', 'terrace_area', 'garden', 'garden_area', 'land_surface',
           'facade_count', 'swimming_pool', 'building_condition', 'price_x_m2', 
           # 'location_lat', 'location_lon',
           'location'])
    df = df.rename(columns={'location_lat': 'lat'})
    df = df.rename(columns={'location_lon': 'lon'})
    df.reset_index(drop=True)
    
    return df

def classification_by_type(df):
    """
    Parameters
    ----------
    df : dataframe
        after have been cleaned we filter the data for each possible type.

    Returns
    -------
    df_houses : dataframe
        dataframe with all the houses in Belgium.
    df_office : dataframe
        dataframe with all the offices in Belgium.
    df_industry : dataframe
        dataframe with all the industry buildings in Belgium.
    df_apartment : dataframe
        dataframe with all the apartment in Belgium.

    """
    df_houses = df.loc[df['type']=='HOUSE']
    df_office = df.loc[df['type']=='OFFICE']
    df_industry = df.loc[df['type']=='INDUSTRY']
    df_apartment = df.loc[df['type']=='APARTMENT']
    return df_houses, df_office, df_industry, df_apartment

def classification_by_region(df):
    """
    Parameters
    ----------
    df : dataframe.
        After that the dataframe in clasiffied by type. We can separated 
        the dataframes by regions in each case.

    Returns
    -------
    df_brussels : dataframe
        This will be the type of buildings input but located in Brussels.
    df_flanders : dataframe
        This will be the type of buildings input but located in Flanders.
    df_wallonie : dataframe
        This will be the type of buildings input but located in Wallonie.

    """
    df_brussels = df.loc[df['region']=='Brussels']
    df_flanders = df.loc[df['region']=='Flanders']
    df_wallonie = df.loc[df['region']=='Wallonie']
    return df_brussels, df_flanders, df_wallonie


def create_df_plot(X, y, name: str):
    """
    To be able to create the differents plots using functions
    It is request to have a dataframe as input.
    In this case both inputs require to have the same len()
    Also they must be columns. But this part is normal because we will need it
    as columns to proccess in the regressions models.

    Parameters
    ----------
    X : array
        Array column that will be in horizontal axis.
    y : array
        Array column that will be in vertical axis.
    name : str
        This is the name that will help to clasify the values in the plot.

    Returns
    -------
    df_plot : dataframe
        dataframe that will use to plot.

    """
    df_plot= pd.concat([
                pd.DataFrame(X, columns=['X']),
                pd.DataFrame(y, columns=['y'])], axis =1)
    df_plot['legend'] = name
    
    return df_plot

def prices_close_to_area(df_data, value, tolerance=20):
    if len(df_data.loc[df_data['area'] == value]) > 0:
        df_prices = df_data.loc[df_data['area'] == value]
        mean_real_price = df_prices.actual_price.mean()
    elif len(df_data.loc[df_data['area'] == value]) == 0:
        df_prices = df_data.loc[(value-tolerance) < df_data['area']]
        df_prices = df_prices.loc[df_prices['area'] < (value+tolerance)]        
        mean_real_price = df_prices.actual_price.mean()
    return np.round(mean_real_price,2)
