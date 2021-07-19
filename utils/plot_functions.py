# -*- coding: utf-8 -*-
"""
@author: HZU
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px




# def show_plot(kind: str):
#     st.write(kind)
#     if kind == "matplotlib":
#         plot = matplotlib_plot
#         st.pyplot(plot)
#     elif kind == "plotly":
#         plot = plotly_plot
#         st.plotly_chart(plot, use_container_width=True)


def matplotlib_plot(chart_type: str, df_plot, label_X: str, label_y: str, title: str):
    """ return matplotlib plots
        df_plot is the dataframe that contain the array X & y 
        and if it is the case also Z
        chart_type == "Scatter", "Line", "3D Scatter" 
        st.echo() is a function that print the code below in the website.        
        
        IMPORTANT:
            To plot use:
                plot = matplotlib_plot(...)
                st.pyplot(plot)
    """
    X = df_plot.X
    y = df_plot.y
    if 'z' in df_plot:
        z = df_plot.z
    else:
        df_plot['z'] = 0
        z = df_plot.z    
        
    fig, ax = plt.subplots()
    if chart_type == "Scatter":
        # with st.echo():
        ax.scatter(x=df_plot.X, y=df_plot.y)
        plt.title(title)
        plt.xlabel(label_X)
        plt.ylabel(label_y)

    elif chart_type == "Line":
        # with st.echo():
        ax.plot(df_plot.X, df_plot.y)
        plt.title(title)
        plt.xlabel(label_X)
        plt.ylabel(label_y)
            
    elif chart_type == "3D_Scatter":
        ax = fig.add_subplot(projection="3d")
        # with st.echo():
        ax.scatter3D(
            xs=df_plot.X,
            ys=df_plot.y,
            zs=z,
        )
        ax.set_xlabel(label_X)
        ax.set_ylabel(label_y)
        ax.set_zlabel("")
        plt.title(title)
        """
        I dont think we will need the next two, in any case, they are created.
        """
    # elif chart_type == "Histogram":
    #     with st.echo():
    #         plt.title(title)
    #         ax.hist(X)
    #         plt.xlabel(label_X)
    #         plt.ylabel(label_y)
            
    # elif chart_type == "Bar":
    #     with st.echo():
    #         ax.bar(x=X, height=y)
    #         plt.title(title)
    #         plt.xlabel(label_X)
    #         plt.ylabel(label_y)

    return fig


def plotly_plot(chart_type: str, df_plot, label_X: str, label_y: str, title: str):
    """ return plotly plots
        df_plot is the dataframe that contain the array X & y 
        and if it is the case also Z
        chart_type == "Scatter", "Line", "3D Scatter" 
        st.echo() is a function that print the code below in the website.
        
        IMPORTANT:
            To plot use:
                plot = plotly_plot
                st.plotly_chart(plot, use_container_width=True)      
    """
    X = df_plot.X
    y = df_plot.y
    if 'z' in df_plot:
        z = df_plot.z
    else:
        df_plot['z'] = 0
        z = df_plot.z    


    if chart_type == "Scatter":
        # with st.echo():
        fig = px.scatter(
            data_frame=df_plot,
            x='X',
            y='y',
            color="legend",
            labels={
                 'X': label_X,
                 'y': label_y
             },                
            title=title,
        )

    elif chart_type == "Line":
        # with st.echo():
        fig = px.line(
            data_frame=df_plot,
            x=df_plot.X,
            y=df_plot.y,
            title=title,
            labels={
                 'X': label_X,
                 'y': label_y
             }
        )
            
    elif chart_type == "3D_Scatter":
        # with st.echo():
        fig = px.scatter_3d(
            data_frame=df_plot,
            x=X,
            y=y,
            z=z,
            title=title,
            labels={
                 'X': label_X,
                 'y': label_y
             }                
        )

        """
        I dont think we will need the next two, in any case, they are created.

        """

    # elif chart_type == "Histogram":
    #     with st.echo():
    #         fig = px.histogram(
    #             data_frame=df_plot,
    #             x=X,
    #             title=title,
    #         )
    # elif chart_type == "Bar":
    #     with st.echo():
    #         fig = px.histogram(
    #             data_frame=df_plot,
    #             x=X,
    #             y=y,
    #             title=title,
    #             histfunc="avg",
    #         )

    return fig

























