# -*- coding: utf-8 -*-
"""
@author: HZU
"""
import pandas as pd



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

    #son 12128
    df = df.dropna(subset=['actual_price'])
    #Ahora son 10795
    df = df.dropna(subset=['area'])
    #Ahora son 9543
    # df = df.dropna(subset=['building_condition'])
    df = df.dropna(subset=['location_lat']) 
    df = df.dropna(subset=['location_lon'])     
    #Ahora son 7059
    df = df.drop_duplicates(subset=['prop_id'])
    #Ahora son 4605
    df = df.fillna(0)
    #Now we add the price by m2
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
    #This are the houses.
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