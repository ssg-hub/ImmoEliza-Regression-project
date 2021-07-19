# -*- coding: utf-8 -*-
"""
@author: HZU
"""
#To create the app
import streamlit as st
import time

#To proccess the data
import pandas as pd
import numpy as np
from utils.data_base_processing import (cleaning_data, classification_by_type,
                                        classification_by_region, create_df_plot)

#To create the models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

#To visualize the plots
import matplotlib.pyplot as plt
import plotly.express as px
from utils.plot_functions import matplotlib_plot, plotly_plot

#For linear model processing
from utils.linear_regression import (data_selection_linear_regression,
                                     traing_and_test_linear_regression
                                     )


#Loading the data in df
df = pd.read_csv('data_real_state_analysis.csv')

#cleaning the dataframe
df = cleaning_data(df)

#Creating the app page, title and setting the columns
#when the columns are set, it works until the last column is called.
#after that it return to be only one column.

st.set_page_config(layout="wide")
st.title('Real estate analysis in Belgium')
st.header('Selection of region / type')
col1, col_mid, col2 = st.beta_columns((2, 0.4, 2))


#first column is called
#here we will do filtering of the data base by type and region
#using a select box and we will print below the dataframe

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

    df_data = df_data.reset_index(drop=True)            
    st.dataframe(df_data)

#Here we will plot the locations for all the rows selected in a map.

with col2:
    st.map(df_data)       


#Here we will select the type of regression that will be applied.


st.markdown("""---""")
col1, col2, col3 = st.beta_columns(3)
col2.header(' A title here')
st.write('\n')
selection_model = col2.selectbox('Please select the model to apply:',
                               ('none',
                                'Elastic Net Regression',
                                'Lasso Regression',
                                'Linear Regression',
                                'Nearest Neighbors Regression',
                                'Ridge Regression',))



st.markdown("""---""")
st.header('Extraction of Target and Features')
st.write('\n')
st.write('In this case I am taking only price as Target and Area as Feature (We need to decide if this is variable or not)')
st.markdown("""---""")

col1, col_mid, col2 = st.beta_columns((1, 0.4, 1))


#Here we will select and plot the test and traning data for the regression selected

if selection_model == 'none':
    st.write('Please select a model that can be used.')

elif selection_model == 'Elastic Net Regression':
    st.write('Please select a model that can be used.')
    
elif selection_model == 'Lasso Regression':
    st.write('Please select a model that can be used.')
    
elif selection_model == 'Linear Regression':
    with col1:
        col1.subheader('We create two arrays')
        #Creating the feature and target data from selection
        X, y = data_selection_linear_regression(df_data)    
        
        #creating a dataframe to plot and ploting the data
        df_plot = create_df_plot(X, y, 'Orginal data')
        plot = plotly_plot('Scatter', df_plot, 'area', 'price', 'Plot of  Target vs Feature (all data)')
        st.plotly_chart(plot, use_container_width=True)        

    with col2:
        col2.subheader('We separate the data into two parts, one to create the model and the other to test the model.')

        #splitting the data in test and training 
        X_train, X_test, y_train, y_test = traing_and_test_linear_regression(X,y)
    
        #creating a dataframe for each type of target and feature
        df_plot_train = create_df_plot(X_train, y_train, 'Training data')
        df_plot_test = create_df_plot(X_test, y_test, 'Test data')

        #concat the two dataframes to be able to plot it together    
        df_plot = [df_plot_train, df_plot_test]
        df_plot = pd.concat(df_plot)
        
        #ploting the data
        plot = plotly_plot('Scatter', df_plot, 'area', 'price', 'Test and training data')    
        st.plotly_chart(plot, use_container_width=True)    

    #Creating a line to separated the selection with the results
    st.markdown("""---""")
    st.header('Testing the LinearRegression model')
    st.markdown("""---""")
    
    col1, col_mid, col2 = st.beta_columns((2, 0.4, 2))

    with col1:
        st.subheader('We create linear regression object')
        st.write('\n')
        st.code("""
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    regressor.score(X_train, y_train)
                """)
        st.write('\n')
        regressor = LinearRegression()
        regressor.fit(X_train, y_train)
        st.write('The regressor score is:', np.round(regressor.score(X_train, y_train), 2))
        st.write('\n')
        st.write('The score of our model with X_test and y_test is:',np.round(regressor.score(X_test, y_test), 2))
        st.write('\n')    
        st.write('Now, we will use the predict method of our model on the test dataset ( X_test )')
        st.write('\n')
    
        df_regressor = create_df_plot(X_test, regressor.predict(X_test), 'Data from model')
        df_plot2 = create_df_plot(X, y, 'Orginal data')
        df_plot2 = [df_regressor, df_plot2]
        df_plot2 = pd.concat(df_plot2)
        
        plot = plotly_plot('Scatter', df_plot2, 'area', 'price', 'Price vs Area training set & org data')
        st.plotly_chart(plot, use_container_width=True)
    
    
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


elif selection_model == 'Nearest Neighbors Regression':
    st.write('Please select a model that can be used.')
    
elif selection_model == 'Ridge Regression':
    st.write('Please select a model that can be used.')
