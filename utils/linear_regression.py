# -*- coding: utf-8 -*-
"""
@author: HZU
"""
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

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




def linear_regression(df):
    
    
    
    
    return