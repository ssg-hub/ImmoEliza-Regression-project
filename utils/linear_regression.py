# -*- coding: utf-8 -*-
"""
@author: HZU
"""
import streamlit as st
import pandas as pd
import numpy as np

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

def data_selection_linear_regression(df_data):
    """
    This function will create the target and feature arrays.
    
    Parameters
    ----------
    df_data : dataframe
        This dataframe is coming form the selection make by the user.
        From this dataframe we will take the target (price) and the feature (area)
    Returns
    -------
    X : array
        This is the feature array that we will use to do the training and test.
    y : array
        This is the target array that we will use to do the training and test.

    """
    with st.echo():    
        X = df_data['area'].to_numpy()
        y = df_data['actual_price'].to_numpy()
        X = X.reshape(len(X),1)
        y = y.reshape(len(y),1)   
    return X, y

def traing_and_test_linear_regression(X,y):
    """
    This function will split the feature and target in test and traning arrays.

    Parameters
    ----------
    X : array
        Feature array.
    y : array
        Target array.

    Returns
    -------
    The different arrays that will be used in the analysis.

    """
    with st.echo():
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=41, test_size=0.2)
    
    return X_train, X_test, y_train, y_test

def linear_regression(df_data):
    col1, col_mid, col2 = st.beta_columns((1, 0.4, 1))
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
        col2.write('\n')         
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
    st.header('Testing the Linear Regression model')
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