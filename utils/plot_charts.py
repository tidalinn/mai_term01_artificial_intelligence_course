import numpy as np
import pandas as pd

from typing import Mapping, Tuple

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
from matplotlib import cm

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_scatter(x, y, title: str):
    
    plt.figure(figsize=(7, 7))

    font_s = 16

    plt.title(f'{title}\n', fontsize=font_s)

    plt.scatter(x, y, c=y, cmap=cm.plasma)

    plt.xlabel('X', fontsize=font_s)
    plt.ylabel('Y', fontsize=font_s)

    plt.grid()
    plt.show()


def plot_gradient_descent_loss_2d(x, y,
                                  weights: list,
                                  title: str,
                                  m_range = np.arange(-30, 60, 2),
                                  b_range = np.arange(-40, 120, 2)):
    
    # dimension
    if x.ndim == 1:
        x = np.array(x).reshape(-1, 1)
    
    
    weights = np.array(weights)
    intercepts, slopes = weights[:, 0], weights[:, 1]
    mse = np.zeros((len(m_range), len(b_range)))
    
    for i, slope in enumerate(m_range):
        for j, intercept in enumerate(b_range):
            mse[i, j] = mean_squared_error(y, x * slope + intercept)
            

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
    fig.add_trace(go.Scatter(x=intercepts,
                             y=slopes,
                             mode='lines',
                             line=dict(width=3),
                             line_color='coral',
                             marker=dict(
                                 opacity=1,
                                 size=np.linspace(19, 1, len(intercepts)),
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


def plot_gradient_descent_function_2d(x, y,
                                      weights: list,
                                      loss_f: Mapping,
                                      title: str,
                                      global_min: list = None):
    
    x_meshed, y_meshed = np.meshgrid(x, y)
    weights = np.array(weights)
    intercepts, slopes = weights[:, 0], weights[:, 1]

    fig = make_subplots(rows=1,
                        subplot_titles=[title])

    
    # contour
    fig.add_trace(go.Contour(z=loss_f(x_meshed, y_meshed),
                             x=x,
                             y=y,
                             colorscale='haline'))


    # descent
    fig.add_trace(go.Scatter(x=intercepts,
                             y=slopes,
                             mode='lines',
                             line=dict(width=3),
                             line_color='coral',
                             marker=dict(
                                 opacity=1,
                                 size=np.linspace(19, 1, len(intercepts)),
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

    
    # global minimum
    if global_min is not None:
        fig.add_trace(go.Scatter(x=[global_min[0]],
                                 y=[global_min[1]],
                                 mode='markers',
                                 marker=dict(size=10, line=dict(width=2)),
                                 marker_color='white',
                                 name='Global Min'))

    
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


def plot_function_2d(x, y,
                     loss_f: Mapping,
                     title: str,
                     weights = None,
                     descent: bool = False,
                     ax = None):

    font_s = 16


    if ax is None:
        # background
        plt.figure(figsize=(8, 6))
        plt.plot(x, loss_f(x, y))


        # axes
        plt.title(f'{title}\n', fontsize=font_s + 2)

        plt.xlabel('w0', fontsize=font_s)
        plt.ylabel('w1', fontsize=font_s)

        plt.grid()

    
    else:
        # background
        ax.plot(x, loss_f(x, y))

        
        # axes
        ax.set_title(f'{title}\n', fontsize=font_s + 2)

        ax.set_xlabel('w0', fontsize=font_s)
        ax.set_ylabel('w1', fontsize=font_s)

        ax.grid()


def plot_gradient_descent_3d(x, y,
                             loss_f: Mapping,
                             title: str,
                             weights = None,
                             descent: bool = False,
                             ax = None):
    
    font_s = 16    
    x_meshed, y_meshed = np.meshgrid(x, y)

    
    # background
    if ax is None:
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(projection='3d')
    

    # function body
    ax.plot_surface(x_meshed, 
                    y_meshed, 
                    loss_f(x_meshed, y_meshed), 
                    alpha=0.6, 
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
    

    # axes
    ax.set_title(title, fontsize=font_s + 2)
    
    ax.set_xlabel('w0', fontsize=font_s)
    ax.set_ylabel('w1', fontsize=font_s)
    ax.set_zlabel('loss', fontsize=font_s)
    
    ax.zaxis.labelpad = 10

    if ax is None:
        plt.show()


def plot_function_gradient_2d_3d(x, y,
                                 loss_f: Mapping,
                                 title: str,
                                 weights = None,
                                 descent: bool = False):
    
    fig = plt.figure(figsize=(16, 6))
    
    
    # first subplot
    ax1 = fig.add_subplot(121)
    plot_function_2d(x, y, loss_f, 
                     f'2D {title} function',
                     weights,
                     descent,
                     ax=ax1)
    
    
    # second subplot
    ax2 = fig.add_subplot(122, projection='3d')
    plot_gradient_descent_3d(x, y, loss_f, 
                             f'3D {title} function',
                             weights,
                             descent,
                             ax=ax2)