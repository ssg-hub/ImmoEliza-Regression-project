# -*- coding: utf-8 -*-
"""

"""
#To create the app
import streamlit as st
import time

#To proccess the data
import pandas as pd
import numpy as np
from utils.data_base_processing import (fixing_lon_lat, cleaning_data, classification_by_type,
                                        classification_by_region, create_df_plot, outliers)

#To create the models
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

#To manage errors
from sklearn.metrics import mean_absolute_error


#To visualize the plots
import matplotlib.pyplot as plt
import plotly.express as px
from utils.plot_functions import matplotlib_plot, plotly_plot

#For linear model processing
from utils.linear_regression import (data_selection_linear_regression,
                                      traing_and_test_linear_regression,
                                      linear_regression
                                      )
# #For Neigbors model processing
# from utils.neighbors_regression import (data_selection_neighbors_regression,
#                                      traing_and_test_neighbors_regression,
#                                      neighbors_regression
#                                      )


#Loading the data in df
df = pd.read_csv('clean_dataframe.csv')


#Creating the app page, title and setting the columns
#when the columns are set, it works until the last column is called.
#after that it return to be only one column.


with st.sidebar:
    # st.set_page_config(layout="wide")
    st.set_page_config(layout="wide")
    st.sidebar.title('Real estate analysis in Belgium')
    st.sidebar.header('Selection of region / type')

    #here we will do filtering of the data base by type and region
    #using a select box and we will print below the dataframe
    
    selection_region = st.sidebar.selectbox('Please select a region in Belgium:',
                                      ('Brussels', 'Flanders', 'Wallonia'))
    selection_type = st.sidebar.selectbox('Please select the building type:',
                                      ('Apartment', 'House','Office','Industry'))    

    selection_outliers = st.sidebar.selectbox('Please select an outlier strategy:',
                                              ('Do not remove', 'Remove'))    

    st.markdown("""---""")    
    value_expand_seleccion_data = False
    if st.button('Data selection'):
        value_expand_seleccion_data = True

    value_expand_linear_regression = False
    if st.button('Linear regression'):
        value_expand_linear_regression = True
    
    value_expand_neighborgs_regression = False
    if st.button('Neighborgs regression'):
        value_expand_neighborgs_regression = True

    value_expand_random_forest_regression = False
    if st.button('Random Forest regression'):
        value_expand_random_forest_regression = True


    value_expand_all_regression = False
    if st.button('All the regression'):
        value_expand_all_regression = True


df = outliers(df, selection_outliers)              

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

st.markdown("""---""")
col1, col_mid, col2 = st.beta_columns((1, 0.1, 1))
col1.write('This is the dataframe result of the filter:')
col1.write('\n')
col1.dataframe(df_data)
# #Here we will plot the locations for all the rows selected in a map.
col2.map(df_data)       
st.markdown("""---""")



expander_seleccion_data = st.beta_expander("Data seleccion", expanded=value_expand_seleccion_data)
with expander_seleccion_data:
    col1, col_mid, col2 = st.beta_columns((1, 0.1, 1))
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

    
expander_linear_regression = st.beta_expander("Linear regression", expanded=value_expand_linear_regression)
with expander_linear_regression:
    col1, col_mid, col2 = st.beta_columns((1, 0.1, 1))    
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
        a_line = float(min(X))
        b_line = float(max(X))
        
        value_selected_linear = st.slider('Area_linear:', a_line, b_line, value=float(100))
        
        st.write('\n')  
        
        z = [float(value_selected_linear)]
        z = np.array(z).reshape(1, -1)
        st.write('The approximate price of the property with that area is :', regressor.fit(X_train, y_train).predict(z))


expander_neighborgs_regression = st.beta_expander("Neighbors regression", expanded=value_expand_neighborgs_regression)
with expander_neighborgs_regression:
    col1, col_mid, col2 = st.beta_columns((1, 0.1, 1))    
    with col1:
        st.subheader('We create Neighbors regression object')
        st.write('\n')
        st.write('Please select the number of neighbors')
        value_selected = st.slider('', 1, 10,4)
        
        st.code("""
        pipe = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors= 'value'))
                ])
        pred = pipe.fit(X_train, y_train).predict(X_test)
                """)
        st.write('\n')
    
        pipe = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=value_selected))
                ])
        pred = pipe.fit(X_train, y_train).predict(X_test)
    
        
        st.write('The pipe score is:', np.round(pipe.score(X_train, y_train), 2))
        st.write('\n')
        st.write('The score of our model with X_test and y_test is:',np.round(pipe.score(X_test, y_test), 2))
        st.write('\n')    
        st.write('Now, we will use the predict method of our model on the test dataset ( X_test )')
        st.write('\n')
    
        df_regressor = create_df_plot(X_test, pipe.predict(X_test), 'Data from model')
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
        a_neig = float(min(X))
        b_neig = float(max(X))
        
        value_selected_neig = st.slider('Area_neighbors:', a_neig, b_neig, value=float(100))
        
        st.write('\n')  
        
        z = [float(value_selected_neig)]
        z = np.array(z).reshape(1, -1)
        st.write('The approximate price of the property with that area is :', pipe.fit(X_train, y_train).predict(z))

expander_random_forest_regression = st.beta_expander("Random Forest Regression", expanded=value_expand_random_forest_regression)
with expander_random_forest_regression:
    col1, col_mid, col2 = st.beta_columns((1, 0.1, 1))
    with col1:
        st.subheader('We create Random Forest regression object')
        st.write('\n')
        st.write('Please select the sample threshold for splitting nodes')
        value_selected = st.slider('', 2, 20, 10)

        st.code("""
pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestRegressor(min_samples_split= 'value',
                                            max_features='log2',
                                            random_state=42))
        ])
pred = pipe.fit(X_train, y_train).predict(X_test)
                """)
        st.write('\n')

        pipe_forest = Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestRegressor(min_samples_split=value_selected, max_features='log2', random_state=42))
        ])
        pred_forest = pipe_forest.fit(X_train, y_train).predict(X_test)

        st.write('The pipe score is:', np.round(pipe_forest.score(X_train, y_train), 2))
        st.write('\n')
        st.write('The score of our model with X_test and y_test is:', np.round(pipe_forest.score(X_test, y_test), 2))
        st.write('\n')
        st.write('Now, we will use the predict method of our model on the test dataset ( X_test )')
        st.write('\n')

        df_regressor = create_df_plot(X_test, pipe_forest.predict(X_test), 'Data from model')
        df_plot2 = create_df_plot(X, y, 'Orginal data')
        df_plot2 = [df_regressor, df_plot2]
        df_plot2 = pd.concat(df_plot2)

        plot = plotly_plot('Scatter', df_plot2, 'area', 'price', 'Price vs Area training set & org data')
        st.plotly_chart(plot, use_container_width=True)

    with col2:
        st.subheader('We now have a model to approximate the value of a property in the location selected')
        st.write('\n')
        st.write('Please choose an area in [m^2] using the slider below to find the approximate value of the property')
        st.write('\n')
        a_forest = float(min(X))
        b_forest = float(max(X))

        value_selected_forest = st.slider('Area_forest:', a_forest, b_forest, value=float(100))

        st.write('\n')

        z = [float(value_selected_forest)]
        z = np.array(z).reshape(1, -1)
        st.write('The approximate price of the property with that area is :', np.round(pipe_forest.fit(X_train, y_train).predict(z),0))

##############################################################################
#############       From here all together

expander_all_together = st.beta_expander("All the regression", expanded=value_expand_all_regression)

with expander_all_together:
    col1, col_mid, col2, col_mid, col3 = st.beta_columns((1, 0.1, 1, 0.1, 1))    
    with col1:
        st.subheader('Linear regression')
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
        st.subheader('Neighbors regression')
        st.write('\n')
        st.write('Please select the number of neighbors')
        value_selected_all_neig = st.slider('Neighbors', 1, 10,4)
        st.write('\n')
    
        pipe_neig = Pipeline([
                    ("scale", StandardScaler()),
                    ("model", KNeighborsRegressor(n_neighbors=value_selected_all_neig))
                ])
        pred_neig = pipe_neig.fit(X_train, y_train).predict(X_test)
    
        
        st.write('The pipe score is:', np.round(pipe_neig.score(X_train, y_train), 2))
        st.write('\n')
        st.write('The score of our model with X_test and y_test is:',np.round(pipe_neig.score(X_test, y_test), 2))
        st.write('\n')    
        st.write('Now, we will use the predict method of our model on the test dataset ( X_test )')
        st.write('\n')
    
        df_regressor = create_df_plot(X_test, pipe_neig.predict(X_test), 'Data from model')
        df_plot2 = create_df_plot(X, y, 'Orginal data')
        df_plot2 = [df_regressor, df_plot2]
        df_plot2 = pd.concat(df_plot2)
        
        plot = plotly_plot('Scatter', df_plot2, 'area', 'price', 'Price vs Area training set & org data')
        st.plotly_chart(plot, use_container_width=True)
    
    with col3:
        st.subheader('Random Forest regression')
        st.write('\n')
        st.write('Please select the sample threshold for splitting nodes')
        value_selected_random_forest = st.slider('Random Forest', 2, 20, 10)

        pipe_forest = Pipeline([
            ("scale", StandardScaler()),
            ("model", RandomForestRegressor(min_samples_split=value_selected_random_forest, max_features='log2', random_state=42))
        ])
        pred = pipe_forest.fit(X_train, y_train).predict(X_test)

        st.write('The pipe score is:', np.round(pipe_forest.score(X_train, y_train), 2))
        st.write('\n')
        st.write('The score of our model with X_test and y_test is:', np.round(pipe_forest.score(X_test, y_test), 2))
        st.write('\n')
        st.write('Now, we will use the predict method of our model on the test dataset ( X_test )')
        st.write('\n')

        df_regressor = create_df_plot(X_test, pipe_forest.predict(X_test), 'Data from model')
        df_plot2 = create_df_plot(X, y, 'Orginal data')
        df_plot2 = [df_regressor, df_plot2]
        df_plot2 = pd.concat(df_plot2)

        plot = plotly_plot('Scatter', df_plot2, 'area', 'price', 'Price vs Area training set & org data')
        st.plotly_chart(plot, use_container_width=True)        
        

    col1, col_area_all_regre, col3 = st.beta_columns((0.5, 1, 0.5))
    
    with col_area_all_regre:
        st.subheader('We will test all the models at the same time')
        st.write('\n')
        st.write('Please chosse an area in [m^2] using the slider below to find the approximate value of the property')
        st.write('\n')
        a_all_reg = float(min(X))
        b_all_reg = float(max(X))
        
        value_all_reg = st.slider('Area_all_neighbors:', a_all_reg, b_all_reg, value=float(100))                

        st.write('\n')  

    col_linear, col_mid, col_neig, col_mid, col_forest = st.beta_columns((1, 0.1, 1, 0.1, 1))        

    with col_linear:
        z_linear = [float(value_all_reg)]
        z_linear = np.array(z_linear).reshape(1, -1)
        st.write('The linear regression approximate the price of the property')
        st.write('with that area to :')
        currency = np.round(regressor.fit(X_train, y_train).predict(z_linear)[0][0], 2)
        currency = "€{:,.2f}".format(currency)
        st.subheader(currency)

    with col_neig:
        z_neig = [float(value_all_reg)]
        z_neig = np.array(z_neig).reshape(1, -1)
        st.write('The neighbors regression approximate the price')
        st.write('of the property with that area to :')
        currency = np.round(pipe_neig.fit(X_train, y_train).predict(z_neig)[0][0], 2)
        currency = "€{:,.2f}".format(currency)
        st.subheader(currency)


    with col_forest:
        z_forest = [float(value_all_reg)]
        z_forest = np.array(z_forest).reshape(1, -1)
        st.write('The Random Forest regression approximate the price')
        st.write('of the property with that area to :')
        currency = np.round(pipe_forest.fit(X_train, y_train).predict(z_forest)[0], 2)
        currency = "€{:,.2f}".format(currency)
        st.subheader(currency)
