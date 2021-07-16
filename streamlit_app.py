# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 12:28:15 2021

@author: HZU
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('data_real_state_analysis.csv')

def cleaning_data(df):
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
    return df

def classification_by_type(df):
    #This are the houses.
    df_houses = df.loc[df['type']=='HOUSE']
    df_office = df.loc[df['type']=='OFFICE']
    df_industry = df.loc[df['type']=='INDUSTRY']
    df_apartment = df.loc[df['type']=='APARTMENT']
    return df_houses, df_office, df_industry, df_apartment

def classification_by_region(df):
    df_brussels = df.loc[df['region']=='Brussels']
    df_flanders = df.loc[df['region']=='Flanders']
    df_wallonie = df.loc[df['region']=='Wallonie']
    return df_brussels, df_flanders, df_wallonie

df = cleaning_data(df)

st.set_page_config(layout="wide")
st.title('Real estate analysis in Belgium')
st.header('Selection of region / type')
col1, col2 = st.beta_columns(2)


with col1:
    selection_region = col1.selectbox('Please select a region in Belgium:',
                                     ('Brussels', 'Flanders', 'Wallonia'))
    selection_type = col1.selectbox('Please select the building type:',
                                     ('Apartment', 'House','Office','Industry'))    

    df_houses, df_office, df_industry, df_apartment = classification_by_type(df)

    if selection_type == 'Apartment':
        df_apt_brus, df_apt_fla, df_apt_wal = classification_by_region(df_apartment)

        if selection_region == 'Brussels':
            df_data = df_apt_brus
        elif selection_region == 'Flanders':
            df_data = df_apt_fla
        elif selection_region == 'Wallonia':
            df_data = df_apt_wal
            
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1) 
        
    elif selection_type == 'House':
        df_h_brus, df_h_fla, df_h_wal = classification_by_region(df_houses)

        if selection_region == 'Brussels':
            df_h_brus = df_h_brus.drop(df_h_brus.loc[df_h_brus['area']>2500].index)
            df_data = df_h_brus
        elif selection_region == 'Flanders':
            df_data = df_h_fla
        elif selection_region == 'Wallonia':
            df_data = df_h_wal
            
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)        
        
    elif selection_type == 'Office':
        df_of_brus, df_of_fla, df_of_wal = classification_by_region(df_office)

        if selection_region == 'Brussels':
            df_data = df_of_brus
        elif selection_region == 'Flanders':
            df_data = df_of_fla
        elif selection_region == 'Wallonia':
            df_data = df_of_wal

            
        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)        
    
    elif selection_type == 'Industry':    
        df_ind_brus, df_ind_fla, df_ind_wal = classification_by_region(df_industry)

        if selection_region == 'Brussels':
            df_data = df_ind_brus
        elif selection_region == 'Flanders':
            df_data = df_ind_fla
        elif selection_region == 'Wallonia':
            df_data = df_ind_wal

        my_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.01)
            my_bar.progress(percent_complete + 1)   
            
    st.dataframe(df_data)

with col2:
    st.map(df_data)       

st.markdown("""---""")
st.header('Extraction of Target and Features')
st.write('\n')
st.write('In this case I am taking only price as Target and Area as Feature (We need to decide if this is variable or not)')
st.markdown("""---""")

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader('We create 2 arrays')
    st.write('\n')
    st.write('X = df_data['+'area'+'].to_numpy()')
    st.write('y = df_data['+'actual_price'+'].to_numpy()')
    st.write('X = X.reshape(len(X),1)')
    st.write('y = y.reshape(len(y),1)')
    st.write('\n')

    X = df_data['area'].to_numpy()
    y = df_data['actual_price'].to_numpy()
    
    l = [X.shape]
    X = X.reshape(l[0][0],1)
    l = [y.shape]
    y = y.reshape(l[0][0],1)    
    
    # plt.figure(figsize=(10, 6))
    fig1 = plt.figure()
    plt.scatter(X, y)
    plt.ylabel('price')
    plt.xlabel('area')
    plt.title('Plot of  Target vs Feature (all data)')    
    st.write(fig1)

with col2:
    st.subheader('We separate the data into two parts, one to create the model and the other to test the model.')
    st.write('\n')
    st.write('X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=41, test_size=0.2)')
    st.write('\n')
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=41, test_size=0.2)
    # plt.figure(figsize=(10, 6))
    fig2 = plt.figure()    
    ax2 = plt.scatter(X_train, y_train, label='Data for training the model')
    ax2 = plt.scatter(X_test, y_test, label='Data to test the model')
    ax2 = plt.title('Plot of Target vs Feature (test & training data)')  
    st.write('\n')
    ax2 = plt.ylabel('price')
    ax2 = plt.xlabel('area')
    ax2 = plt.legend()
    st.write(fig2)

st.markdown("""---""")
st.header('Testing a model (LinearRegression)')
st.markdown("""---""")
st.write('\n')

col1, col2 = st.beta_columns(2)

with col1:
    st.subheader('We create linear regression object')
    st.write('\n')
    st.write('regressor = LinearRegression()')
    st.write('regressor.fit(X_train, y_train)')
    st.write('regressor.score(X_train, y_train)')
    st.write('\n')
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    st.write('The regressor score is:', regressor.score(X_train, y_train))
    st.write('\n')
    st.write('The score of our model with X_test and y_test is:',regressor.score(X_test, y_test))
    st.write('\n')    
    st.write('Now, we will use the predict method of our model on the test dataset ( X_test )')
    st.write('\n')
    fig3 = plt.figure()
    plt.scatter(X_test, regressor.predict(X_test), label='Data from model')
    plt.scatter(X, y, label='Orginal data')
    plt.xlabel('Area')
    plt.ylabel('Price')
    plt.title('Price vs Area (Training set)')
    st.write(fig3)


with col2:
    st.subheader('We now have a model to approximate the value of a property in the location selected')
    st.write('\n')
    st.write('Please chosse an area in [m^2] using the slider below to find the approximate value of the property')
    st.write('\n')
    a = float(min(X))
    b = float(max(X))
    
    value_selected = st.slider('Area:', a, b)
    
    st.write('\n')  
    
    z = [float(value_selected)]
    z = np.array(z).reshape(1, -1)
    st.write('The approximate price of the property with that area is :', regressor.fit(X_train, y_train).predict(z))



































