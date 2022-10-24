import numpy as np
import pandas as pd

from typing import Mapping, Tuple

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib import cm

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_data(features, target, 
              names: dict,
              fit_line = None, 
              x_range: Tuple = None, 
              y_range: Tuple = None, 
              dx: int = 20, 
              dy: int = 20):

    # figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=features, 
                             y=target, 
                             mode='markers', 
                             marker=dict(size=6), 
                             name='data'))

    # regression
    if fit_line is not None:
        fig.add_trace(
            go.Scatter(
                x=features,
                y=fit_line,
                line_color='red',
                mode='lines',
                line=dict(width=5),
                name='Fitted line',
            )
        )
    
    # figure
    fig.update_layout(
        width=500,
        height=500,
        title=names['title'],
        title_x=0.5,
        title_y=0.93,
        xaxis_title=names['x'],
        yaxis_title=names['y'],
        margin=dict(t=60),
    )

    # axes
    if x_range is not None and y_range is not None:
        fig.update_xaxes(range=x_range, tick0=x_range[0], dtick=dx)
        fig.update_yaxes(range=y_range, tick0=y_range[0], dtick=dy)

    return fig


def plot_gradient_descent_2d(features, target,
                             weights: list,
                             title: str,
                             m_range = np.arange(-30, 60, 2),
                             b_range = np.arange(-40, 120, 2),
                             step_size: int = 1):
    
    # dimension
    if features.ndim == 1:
        x = np.array(features).reshape(-1, 1)
    
    
    weights = np.array(weights)
    intercepts, slopes = weights[:, 0], weights[:, 1]
    mse = np.zeros((len(m_range), len(b_range)))
    
    for i, slope in enumerate(m_range):
        for j, intercept in enumerate(b_range):
            mse[i, j] = mean_squared_error(target, features * slope + intercept)
            

    # background
    fig = make_subplots(rows=1,
                        subplot_titles=[title])
    

    # function
    fig.add_trace(go.Contour(z=mse,
                             x=b_range, 
                             y=m_range, 
                             name='', 
                             colorscale='haline'))
    

    # gradient
    fig.add_trace(go.Scatter(x=intercepts[::step_size],
                             y=slopes[::step_size],
                             mode='lines',
                             line=dict(width=3),
                             line_color='coral',
                             marker=dict(
                                 opacity=1,
                                 size=np.linspace(19, 1, len(intercepts[::step_size])),
                                 line=dict(width=2)),
                             name='Descent'))
    
    
    # start point
    fig.add_trace(go.Scatter(x=[intercepts[0]],
                             y=[slopes[0]],
                             mode='markers',
                             marker=dict(size=20, line=dict(width=2)),
                             marker_color='orangered',
                             name='Start'))
    
    
    # end point
    fig.add_trace(go.Scatter(x=[intercepts[-1]],
                             y=[slopes[-1]],
                             mode='markers',
                             marker=dict(size=20, line=dict(width=2)),
                             marker_color='yellowgreen',
                             name='End'))
    
    
    # layout
    fig.update_layout(
        width=700,
        height=600,
        margin=dict(t=60),
        legend=dict(yanchor='top', y=0.99, xanchor='left', x=0.01),
    )
    
    
    # axes
    fig.update_xaxes(
        title='w0',
        range=[b_range.min(), b_range.max()],
    )
    
    fig.update_yaxes(
        title='w1',
        range=[m_range.min(), m_range.max()],
    )
    
    return fig


def plot_gradient_descent_function_2d(x_meshed, y_meshed,
                                      x_axis, y_axis,
                                      weights: list,
                                      loss_f: Mapping,
                                      title: str,
                                      step_size: int = 1):
    
    weights = np.array(weights)
    intercepts, slopes = weights[:, 0], weights[:, 1]

    fig = make_subplots(rows=1,
                        subplot_titles=[title])

    
    # contour
    fig.add_trace(go.Contour(z=loss_f(x_meshed, y_meshed),
                             x=x_axis,
                             y=y_axis,
                             colorscale='haline'))


    # descent
    fig.add_trace(go.Scatter(x=intercepts[::step_size],
                             y=slopes[::step_size],
                             mode='lines',
                             line=dict(width=3),
                             line_color='coral',
                             marker=dict(
                                 opacity=1,
                                 size=np.linspace(19, 1, len(intercepts[::step_size])),
                                 line=dict(width=2)),
                             name='Descent'))

    
    # start point
    fig.add_trace(go.Scatter(x=[intercepts[0]],
                             y=[slopes[0]],
                             mode='markers',
                             marker=dict(size=20, line=dict(width=2)),
                             marker_color='orangered',
                             name='Start'))


    # end point
    fig.add_trace(go.Scatter(x=[intercepts[-1]],
                             y=[slopes[-1]],
                             mode='markers',
                             marker=dict(size=20, line=dict(width=2)),
                             marker_color='yellowgreen',
                             name='End'))

    
    # legend
    fig.update_layout(width=700,
                      height=600,
                      margin=dict(t=60),
                      legend=dict(yanchor='top', y=0.99, 
                                  xanchor='left', x=0.01))

    
    # axes
    fig.update_xaxes(title='w0')
    fig.update_yaxes(title='w1')

    return fig


def plot_gradient_descent_3d(x_meshed, y_meshed,
                             loss_f: Mapping,
                             title: str,
                             weights = None,
                             descent: bool = False):
    
    font_s = 16    

    # background
    fig = plt.figure(figsize = (15, 10))
    ax = fig.add_subplot(projection = '3d')
    

    # function body
    ax.plot_surface(x_meshed, 
                    y_meshed, 
                    loss_f(x_meshed, y_meshed), 
                    alpha = 0.6, 
                    cmap=cm.coolwarm)
    

    if descent == True:

        weights = np.array(weights)
        intercepts, slopes = weights[:, 0], weights[:, 1]
        
        start_point = intercepts[0], slopes[0]
        end_point = intercepts[-1], slopes[-1]


        # descent
        ax.plot(intercepts, 
                slopes, 
                loss_f(intercepts, slopes),
                color='blue')
        
        # start point
        ax.scatter(*start_point,
                loss_f(*start_point), 
                color='red', 
                s=300)
        
        # end point
        ax.scatter(*end_point,
                   loss_f(*end_point),
                   color='coral',
                   marker='*',
                   s=300)
    

    ax.set_title(title, fontsize=font_s + 2)
    
    ax.set_xlabel('x', fontsize=font_s)
    ax.set_ylabel('y', fontsize=font_s)
    ax.set_zlabel('loss', fontsize=font_s)
    
    ax.zaxis.labelpad = 10

    plt.show()