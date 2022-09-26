# plot_charts module
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Mapping, Tuple


# PLOT_2D_chart
def plot_2d_chart(x: np.array, 
                  y: np.array, 
                  main_title: str,
                  fig_size: Tuple = (8, 6)):
    """ Plot 2D chart

    Args:
        x (np.array): X axis values
        y (np.array): Y axis values
        main_title (str): graph title
        fig_size (Tuple, optional): plot size. Defaults to (8, 6)
    
    Result:
       2D chart
    """

    try:
        plt.figure(figsize=fig_size)
        
        plt.plot(x, y)

        plt.title(main_title, fontsize=16)
        plt.xlabel('X', fontsize=16)
        plt.ylabel('Y', fontsize=16)

        plt.grid()
        plt.show()

    except:
        print('Убедитесь в корректности переданных аргументов')
    

# PLOT_ANIMATED_2D_CHART
def plot_animated_2d_chart(x: np.array, 
                           y: np.array, 
                           grads_r: np.array, 
                           loss_f: Mapping, 
                           deriv_f: Mapping, 
                           main_title: str,
                           x_range: list, 
                           y_range: list):

    """ Plot animated 2D chart

    Args:
        x (np.array): X axis values
        y (np.array): Y axis values (loss function results)
        grads_r (np.array): collection of gradients
        loss_f (Mapping): loss function
        deriv_f (Mapping): derivative function
        main_title (str): graph title
        x_range (list, optional): X axis scaling. Defaults to [x.min(), x.max()]
        y_range (list, optional): Y axis scaling. Defaults to [y.min(), y.max()]
    
    Result:
        animated 2d chart
    """
    
    try:
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        x_min -= 0.25
        x_max += 0.25
        y_min -= 0.25
        y_max += 0.25

        s = grads_r
        xx = s
        yy = loss_f(s)

        # velocity = (vx, vy)
        vx = deriv_f(s)
        vy = -deriv_f(s)  

        speed = np.sqrt(vx ** 2 + vy ** 2)

        ux = vx / speed
        uy = 0

        # end coords for the unit tangent vector at (xx, yy)
        xend = xx + ux + 10
        yend = yy + uy

        # end coords for the unit normal vector at (xx, yy)
        xnoe = xx - uy
        ynoe = yy + ux + 10

        
        fig = go.Figure(
            # data
            data = [go.Scatter(x = x, y = y,
                            name = 'gradient descent',
                            mode = 'lines',
                            line = dict(width = 2, color = 'blue')),

                    go.Scatter(x = x, y = y,
                                name = 'gradient',
                                mode = 'lines',
                                line = dict(width = 2, color = 'blue')),

                    go.Scatter(x = x, y = y,
                                name = 'X, Y parallels',
                                mode = 'lines',
                                line = dict(width = 2, color = 'blue')),

                    go.Scatter(x = x, y = y,
                                name = 'initial function',
                                mode = 'lines',
                                line = dict(width = 2, color = 'blue'))],

            # layout
            layout = go.Layout(
                xaxis = dict(range = [x_min, x_max], autorange = False, zeroline = False),
                yaxis = dict(range = [y_min, y_max], autorange = False, zeroline = False),

                title_text = main_title, hovermode='closest',

                updatemenus = [dict(type = 'buttons',
                                    buttons = [dict(label = 'Play',
                                                    method = 'animate',
                                                    args = [None])])]),

            # frames
            frames = [go.Frame(
                data = [go.Scatter(x = xx,
                                y = yy,
                                name = 'gradient descent',
                                mode = 'markers',
                                marker = dict(color='green', size=5)),

                        go.Scatter(x = [xx[k]],
                                y = [yy[k]],
                                name = 'gradient',
                                mode = 'markers',
                                marker = dict(color='red', size=10)),

                    go.Scatter(x = [xx[k], xend[k], None, xx[k], xnoe[k]],
                                y = [yy[k], yend[k], None, yy[k], ynoe[k]],
                                mode = "lines",
                                line = dict(color="#FF8C00", width=2))])

                    for k in range(len(grads_r))]
        )


        fig.update_layout(
            width = 900,
            height = 500,
            xaxis_title = 'X',
            yaxis_title = 'Y'
        )


        fig.show()

    except:
        print('Убедитесь в корректности переданных аргументов')



# PLOT_2D_CONTOURS_CHART
def plot_2d_contours_chart(x: np.array, 
                           y: np.array, 
                           loss_f: Mapping, 
                           coords_s: Tuple = (None), 
                           axes_names: list = ['x', 'y']):

    """ Plot 2D depths chart

    Args:
        x (np.array): 1st axis weights
        y (np.array): 2nd axis weights
        loss_f (Mapping): loss function
        coords_s (Tuple, optional): coords of starting point. Defaults to (None)
        axes_names (list, optional): axes names. Defaults to ['x', 'y']
    
    Result:
        2D depths chart
    """
    
    try:
        fig, ax = plt.subplots(figsize = (8, 8))

        plt.contourf(x, y, loss_f(x, y), cmap = 'Blues')

        ax.scatter(*coords_s, c = 'red', marker = '*', s = 300)
        coords_s = np.array(coords_s) + 0.2
        ax.text(*coords_s, 'A', size = 24)

        ax.scatter(0, 0, c = 'red', marker = '*', s = 300)
        ax.text(-0.1, 0.3, 'B', size = 24)

        ax.set_xlabel(axes_names[0], fontsize = 16)
        ax.set_ylabel(axes_names[1], fontsize = 16)

        plt.grid()
        plt.show()

    except:
        print('Убедитесь в корректности переданных аргументов')



# PLOT_3D_CHART
def plot_3d_chart(x: np.array, 
                  y: np.array, 
                  loss_f: Mapping, 
                  coords_s: Tuple = (None),
                  axes_names: list = ['x', 'y']):

    """ Plot 3D chart

    Args:
        x (np.array): 1st axis weights
        y (np.array): 2nd axis weights
        loss_f (Mapping): loss function
        coords_s (Tuple, optional): coords of starting point. Defaults to (None)
        axes_names (list, optional): axes names. Defaults to ['x', 'y']

    Result:
        3D chart
    """
    
    try:
        fig = plt.figure(figsize = (15, 10))
        ax = fig.add_subplot(projection = '3d')

        ax.plot_surface(x, y, loss_f(x, y), alpha = 0.4, cmap = 'Blues')
        
        if coords_s != (None):
            ax.scatter(*coords_s, c = 'red', marker = '*', s = 200)
            coords_s = np.array(coords_s) + 0.3
            ax.text(*coords_s, 'A', size = 24)

        ax.scatter(0, 0, 0, c = 'red', marker = '*', s = 200)
        ax.text(-0.1, -1.5, 4, 'B', size = 24)
        
        ax.set_xlabel(axes_names[0], fontsize = 16)
        ax.set_ylabel(axes_names[1], fontsize = 16)
        ax.set_zlabel('loss_func({0})'.format(', '.join(axes_names)), fontsize = 16)
        
        plt.show()
    
    except:
        print('Убедитесь в корректности переданных аргументов')



# PLOT_3D_GRADIENT_CHART
def plot_3d_gradient_chart(x: np.array, 
                           y: np.array, 
                           loss_f: Mapping, 
                           coords_s: Tuple,
                           grad_weights: list,
                           axes_names: list = ['x', 'y']):

    """ Plot 3D gradient descent chart

    Args:
        x (np.array): 1st axis weights
        y (np.array): 2nd axis weights
        loss_f (Mapping): loss function
        coords_s (Tuple): coords of starting point
        grad_weights (list): x weights list, y weights list, loss weights list
        axes_names (list, optional): axes names. Defaults to ['x', 'y']

    Result:
        3D gradient descent chart
    """
    
    try:
        fig = plt.figure(figsize = (15, 10))
        ax = fig.add_subplot(projection = '3d')

        ax.plot_surface(x, y, loss_f(x, y), alpha = 0.4, cmap = 'Blues')
        
        coords_s = np.array(coords_s) + 0.3
        ax.text(*coords_s, 'A', size = 24)
        ax.text(-0.1, -1.5, 4, 'B', size = 24)

        ax.plot(*grad_weights, '.-', c = 'red')
        
        ax.set_xlabel(axes_names[0], fontsize = 16)
        ax.set_ylabel(axes_names[1], fontsize = 16)
        ax.set_zlabel('loss_func({0})'.format(', '.join(axes_names)), fontsize = 16)
        
        plt.show()
    
    except:
        print('Убедитесь в корректности переданных аргументов')



# PLOT_ANIMATED_#D_CHART
def plot_animated_3d_chart(x: np.array, 
                           y: np.array, 
                           grad_weights: list,
                           loss_f: Mapping,
                           main_title: str):

    """ Plot 3D animated chart

    Args:
        x (np.array): 1st axis weights
        y (np.array): 2nd axis weights
        grad_weights (list): x weights list, y weights list, loss weights list
        loss_f (Mapping): loss function
        main_title (str): _description_
    
    Result:
        3D animated chart.
    """

    try:
        x_list, y_list, z_list = grad_weights

        fig = go.Figure(
        data = [go.Scatter3d(x = x_list, 
                             y = y_list,
                             z = z_list,
                             mode = 'lines',
                             line = dict(width = 2, color = 'blue')),
                
                go.Scatter3d(x = x_list, 
                             y = y_list,
                             z = z_list,
                             mode = 'lines',
                             line = dict(width = 2, color = 'blue')),
                
                go.Surface(x = x, 
                           y = y, 
                           z = loss_f(x, y),
                           colorscale ='Blues',
                           opacity = 0.5)],
        
        layout = go.Layout(        
            title_text=main_title, hovermode='closest',

            updatemenus = [dict(type = 'buttons',
                                buttons = [dict(label = 'Play',
                                                method = 'animate',
                                                args = [None])])]),
        
        frames = [go.Frame(
            data = [go.Scatter3d(x = x_list,
                                 y = y_list,
                                 z = z_list,
                                 mode = 'lines+markers',
                                 marker = dict(color='green', size=5)),

                    go.Scatter3d(x = [x_list[k]],
                                 y = [y_list[k]],
                                 z = [z_list[k]],
                                 mode = 'markers',
                                 marker = dict(color='red', size=10))])

                for k in range(len(x_list))]
        )


        fig.update_layout(
            width = 900,
            height = 900,
            scene = dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='loss_func(x, y)',
            ),
            showlegend=False
        )
        

        fig.show()

    except:
            print('Убедитесь в корректности переданных аргументов')