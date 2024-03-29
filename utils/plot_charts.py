import numpy as np

from typing import Mapping

from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import cm

import seaborn as sns

from PIL import Image

import imageio

import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_scatter(x, y, title: str):
    
    plt.figure(figsize=(7, 7))

    font_s = 16

    plt.title(f'{title}\n', fontsize=font_s)

    plt.plot(x, y, c=y, cmap=cm.plasma)

    plt.xlabel('X', fontsize=font_s)
    plt.ylabel('Y', fontsize=font_s)

    plt.grid()
    plt.show()


def plot_contour_interactive_dataset_2d(x, y,
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


def plot_contour_interactive_custom_func_2d(x, y,
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


def plot_contour_2d(x, y,
                    loss_f: Mapping,
                    weights: list,
                    title: str,
                    global_min: list = None):
    
    font_s = 16
    
    x_meshed, y_meshed = np.meshgrid(x, y)
    z = loss_f(x_meshed, y_meshed)
    
    
    # plot color
    levels = np.linspace(np.min(z), np.max(z), 20)
    
    
    # descent
    weights = np.array(weights)
    intercepts, slopes = weights[:, 0], weights[:, 1]

    start_point = intercepts[0], slopes[0]
    end_point = intercepts[-1], slopes[-1]
    

    # background
    fig = plt.figure(figsize=(10, 8))
    ax = fig.subplots()
    
    
    # function body
    ax.contourf(x_meshed, 
                y_meshed, 
                z, 
                levels=levels, 
                cmap=cm.coolwarm,
                alpha=0.9)


    # descent
    ax.plot(intercepts, 
            slopes,
            color='orange',
            path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()],
            linewidth=3,
            label='Descent')

    
    # start point
    ax.scatter(*start_point,
               color='red', 
               path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()],
               s=500,
               label='Start',
               zorder=10)

    
    # end point
    ax.scatter(*end_point,
               color='yellow',
               marker='*',
               path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()],
               s=500,
               label='End',
               zorder=10)
    
    
    # global minimum
    if global_min is not None:
        ax.scatter(global_min[0],
                   global_min[1],
                   color='white',
                   path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()],
                   s=200,
                   label='Global Min')

    
    ax.legend(loc='lower right',
              labelspacing=1.6,
              borderpad=1,
              fontsize='large')
        
        
    # axes
    ax.set_title(f'{title}\n', fontsize=font_s + 2)

    ax.set_xlabel('w0', fontsize=font_s)
    ax.set_ylabel('w1', fontsize=font_s)

    plt.show()


def plot_custom_function_3d(x, y,
                            loss_f: Mapping,
                            title: str,
                            weights = None,
                            descent: bool = False,
                            ax = None,
                            global_min: list = None,
                            view: Mapping = None):
    
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
                color='orange',
                path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()],
                linewidth=2,
                label='Descent')
        

        # start point
        ax.scatter(*start_point,
                   loss_f(*start_point), 
                   color='red', 
                   path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()],
                   s=300,
                   label='Start')
        

        # end point
        ax.scatter(*end_point,
                   loss_f(*end_point),
                   color='yellow',
                   marker='*',
                   path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()],
                   s=300,
                   label='End')
        
        
        # legend
        ax.legend(loc='upper right',
                  labelspacing=1.6,
                  borderpad=1,
                  fontsize='large')
        

    # global minimum
    if global_min is not None:
        x_min, y_min = global_min
        
        ax.scatter(x_min,
                   y_min,
                   loss_f(x_min, y_min),
                   color='black',
                   s=300,
                   label='Global Min')
    

    # axes
    if view is True:
        ax.view_init(0, 35)

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
    plot_custom_function_3d(x, y, loss_f, 
                            f'3D {title} function',
                            weights,
                            descent,
                            ax=ax2)


def create_frame_3d(i, x, y, loss_f, grads,
                    title, title_grad, global_min, path):
    
    font_s = 16

    x_meshed, y_meshed = np.meshgrid(x, y)
    x_grads, y_grads = grads[:, 0], grads[:, 1]
    x_min, y_min = global_min[0], global_min[1]


    # background
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(projection='3d')


    # function body
    ax.plot_surface(x_meshed, 
                    y_meshed, 
                    loss_f(x_meshed, y_meshed), 
                    alpha=0.6, 
                    cmap=cm.coolwarm)


    # descent
    ax.scatter(x_grads[i], 
               y_grads[i],
               loss_f(x_grads[i], y_grads[i]),
               color='red', 
               path_effects=[pe.Stroke(linewidth=4, foreground='black'), pe.Normal()],
               s=20,
               label=title_grad)


    # global minimum
    ax.scatter(x_min,
               y_min,
               loss_f(x_min, y_min),
               color='black',
               s=20,
               label='Global Min')


    # axes
    ax.set_title(f'{title}. Step №{i}\n', fontsize=font_s + 2)

    ax.set_xlabel('w0', fontsize=font_s)
    ax.set_ylabel('w1', fontsize=font_s)
    ax.set_zlabel('loss', fontsize=font_s)

    ax.zaxis.labelpad = 10


    plt.savefig(f'{path}/{i}.png', 
                transparent = False,  
                facecolor = 'white')
    plt.close()


def create_frame_2d(i, x, y, loss_f, grads,
                    title, title_grad, global_min, path):
    
    font_s = 16
    
    x_meshed, y_meshed = np.meshgrid(x, y)
    z_meshed = loss_f(x_meshed, y_meshed)
    
    x_grads, y_grads = grads[:, 0], grads[:, 1]
    x_min, y_min = global_min[0], global_min[1]
    
    
    # plot color
    levels = np.linspace(np.min(z_meshed), np.max(z_meshed), 20)
    

    # background
    fig = plt.figure(figsize=(9, 9))
    ax = fig.subplots()
    
    
    # function body
    ax.contourf(x_meshed, 
                y_meshed, 
                z_meshed, 
                levels=levels, 
                cmap=cm.coolwarm,
                alpha=0.9)

    
    # descent
    ax.scatter(x_grads[i], 
               y_grads[i],
               color='red', 
               path_effects=[pe.Stroke(linewidth=5, foreground='black'), pe.Normal()],
               s=20,
               label=title_grad)

    
    # global minimum
    ax.scatter(x_min,
               y_min,
               color='black',
               s=20,
               label='Global Minimum',
               zorder=10)

    
    ax.legend(loc='lower right',
              fontsize='large')
        
        
    # axes
    ax.set_title(f'{title}. Step №{i}', fontsize=font_s + 2)

    ax.set_xlabel('w0', fontsize=font_s)
    ax.set_ylabel('w1', fontsize=font_s)
    
    
    plt.savefig(f'{path}/{i}.png', 
                transparent = False,  
                facecolor = 'white')
    plt.close()


def create_gif(x: np.array, 
               y: np.array,
               loss_f: Mapping,
               grads: np.array,
               title: str,
               title_grad: str,
               title_file: str,
               global_min: list,
               create_frame: Mapping,
               path='./gifs'):
    
    for i in range(grads.shape[0]):
        create_frame(i, x, y, loss_f, grads,
                     title, title_grad, global_min, path)

    frames = [Image.open(f'{path}/{i}.png') for i in range(grads.shape[0])]

    imageio.mimsave(f'{path}/{title_file}.gif', frames, fps=2)


def plot_contour_search_2d(x, y,
                           loss_f: Mapping,
                           pop,
                           title: str,
                           z = None):
    font_s = 16

    x_meshed, y_meshed = np.meshgrid(x, y)

    if z is None:
        z = loss_f(x_meshed, y_meshed)

    
    # background
    fig, ax = plt.subplots(figsize=(10, 7))
    
    
    # function body
    contour_map = ax.contour(x_meshed, 
                             y_meshed, 
                             z, 
                             200,
                             cmap=cm.coolwarm)


    try:
        # search
        ax.scatter(pop.get_x()[:, 0], 
                   pop.get_x()[:, 1], 
                   color='yellow', 
                   path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()],
                   s=10,
                   zorder=10,
                   label='pop')
        
        # global minimum
        x_min, y_min = pop.get_x()[pop.best_idx()]
        ax.scatter(x_min, 
                   y_min, 
                   marker='*',
                   color='red', 
                   path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()],
                   zorder=9,
                   s=200,
                   label='global minimum')
    
    except:            
        # global minimum
        ax.scatter(pop[:, 0], 
                   pop[:, 1], 
                   color='red', 
                   path_effects=[pe.Stroke(linewidth=3, foreground='black'), pe.Normal()],
                   zorder=9,
                   s=10,
                   label='global minimum')

    
    # axes
    ax.set_title(f'{title}\n', fontsize=font_s+2)
    ax.set_xlabel('X', fontsize=font_s)
    ax.set_ylabel('Y', fontsize=font_s)
    
    plt.colorbar(contour_map)
    plt.tight_layout()
    plt.legend(loc='lower left',
               labelspacing=1.6,
               borderpad=1,
               fontsize='large')
    plt.grid()

    plt.show()


def plot_cv_image_landscape_3d(x, y, z,
                               title: str):
    
    font_s = 16    
    x_meshed, y_meshed = np.meshgrid(x, y)

    
    # background
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection='3d')
    

    # function body
    ax.plot_surface(x_meshed, 
                    y_meshed, 
                    z, 
                    alpha=0.6, 
                    cmap=cm.coolwarm,
                    linewidth=0, 
                    antialiased=False)

    ax.set_title(title, fontsize=font_s + 2)
    
    ax.set_xlabel('X', fontsize=font_s)
    ax.set_ylabel('Y', fontsize=font_s)
    ax.set_zlabel('Img', fontsize=font_s)
    
    ax.zaxis.labelpad = 10

    plt.show()


def plot_chart(data: np.array,
               distr_f: Mapping,
               title: str,
               title_x: str = 'x',
               title_y: str = 'f(x)'):
    
    font_s = 16
    plt.figure(figsize=(10, 6))

    plt.scatter(data, distr_f(data))

    plt.title(f'{title}\n', fontsize=font_s + 2)
    plt.xlabel(title_x, fontsize=font_s)
    plt.ylabel(title_y, fontsize=font_s)

    plt.grid()
    plt.show()


def plot_taget_known_distribution_chart(data: np.array,
                                        distr_f: Mapping,
                                        distr_n: Mapping,
                                        title: str,
                                        const: float = 1,
                                        title_x: str = 'x',
                                        title_y: str = 'f(x)'):
    
    if const < 1:
        raise ValueError('const should be greater than 0')
    
    font_s = 16
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data, distr_f(data), label='target distribution')
    ax.plot(data, const * distr_n(data), label='known distribution')
    
    plt.title(f'{title}\n', fontsize=font_s + 2)
    plt.xlabel(title_x, fontsize=font_s)
    plt.ylabel(title_y, fontsize=font_s)
    plt.legend()
    
    plt.grid()
    plt.show()


def plot_sampled_chart(data: np.array,
                       distr_f: Mapping,
                       samples: np.array,
                       title: str,
                       title_x: str = 'x',
                       title_y: str = 'f(x)',
                       **kwargs):

    font_s = 16

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data, distr_f(data), label='target distribution')

    ax.scatter(samples,
               distr_f(samples),
               s=10,
               color='orange',
               label='sample distribution')
    
    ax.hist(samples, 
            bins=80, 
            density=True, 
            alpha=0.2, 
            color='green', 
            edgecolor='black',
            label='sample density')

    sns.kdeplot(samples,
                bw_method=.1, 
                color='green',
                label='sample density line')
    
    for key, value in kwargs.items():
        if key == 'self_samples': 
            ax.hist(value, 
                    bins=80, 
                    density=True, 
                    alpha=0.2, 
                    color='purple', 
                    edgecolor='black',
                    label='self sample density')

            sns.kdeplot(value, 
                        bw_method=.1, 
                        color='purple',
                        label='self density line')
    
    plt.title(f'{title}\n', fontsize=font_s + 2)
    plt.xlabel(title_x, fontsize=font_s)
    plt.ylabel(title_y, fontsize=font_s)
    plt.legend()

    plt.grid()
    plt.show()


def plot_check_dots_chart(data: np.array,
                          distr_f: Mapping,
                          samples: np.array,
                          title: str,
                          title_x: str = 'x',
                          title_y: str = 'f(x)'):
    
    font_s = 16
    x, y, c = samples
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(data, distr_f(data))
    ax.scatter(x, y, c=c, cmap="RdYlGn", lw=0, s=2, alpha=0.8)

    plt.title(f'{title}\n', fontsize=font_s + 2)
    plt.xlabel(title_x, fontsize=font_s)
    plt.ylabel(title_y, fontsize=font_s)

    plt.grid()
    plt.show()


def compare_generated_data(data: np.array, 
                           numpy_generated: np.array, 
                           title: str,
                           hist: bool = False):
    
    font_s = 16
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if hist == False:
        ax.plot(data, 
                alpha=0.5, 
                label='generated data')

        ax.plot(numpy_generated, 
                alpha=0.5,
                label='numpy generated data')
    
    else:
        ax.hist(data, 
                alpha=0.5,
                label='generated data')

        ax.hist(numpy_generated, 
                alpha=0.5, 
                label='numpy generated')
    
    plt.title(f'{title}\n', fontsize=font_s + 2)
    plt.xlabel('loops', fontsize=font_s)
    plt.ylabel('generated', fontsize=font_s)
    
    plt.legend()
    
    plt.grid()
    plt.show()


def compare_density(data: np.array, 
                    numpy_generated: np.array, 
                    title: str,
                    title_x: str = 'loops',
                    title_y: str = 'generated',
                    bins: int = 40):
    
    font_s = 16

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(data, 
            bins=bins, 
            density=True, 
            alpha=0.2, 
            color='blue', 
            edgecolor='black',
            label='generated data density')

    sns.kdeplot(data, 
                bw_method=.1, 
                color='blue',
                label='generated data density line')
    
    ax.hist(numpy_generated, 
            bins=bins, 
            density=True, 
            alpha=0.2, 
            color='orange', 
            edgecolor='black',
            label='numpy generated data density')

    sns.kdeplot(numpy_generated, 
                bw_method=.1, 
                color='orange',
                label='numpy generated data density line')
    
    plt.title(f'{title}\n', fontsize=font_s + 2)
    plt.xlabel(f'{title_x}\n', fontsize=font_s)
    plt.ylabel(f'{title_y}\n', fontsize=font_s)
    plt.legend()
    
    plt.grid()
    plt.show()


def plot_density(generated_data: np.array,
                 title: str,
                 bins: int = 40):
    
    font_s = 16

    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(generated_data, 
            bins=bins, 
            density=True, 
            alpha=0.2, 
            color='blue', 
            edgecolor='black',
            label='generated data density')

    sns.kdeplot(generated_data, 
                bw_method=.1, 
                color='blue',
                label='generated data density line')
    
    plt.title(f'{title}\n', fontsize=font_s + 2)
    plt.xlabel('x', fontsize=font_s)
    plt.ylabel('y', fontsize=font_s)
    plt.legend()
    
    plt.grid()
    plt.show()


def plot_generated_data(x: np.array,
                        y: np.array,
                        title: str):
    
    font_s = 16
    
    plt.figure(figsize=(10, 6))
    
    plt.scatter(x, y)
    
    plt.title(f'{title}\n', fontsize=font_s + 2)
    plt.xlabel('x', fontsize=font_s)
    plt.ylabel('y', fontsize=font_s)
    
    plt.grid()
    plt.show()


def plot_gaussian_mixture(x: np.array,
                          y: np.array,
                          title: str):
    font_s = 16

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x, y[:, 0], label='1st distribution density')
    ax.plot(x, y[:, 1], label='2nd distribution density')

    plt.title(f'{title}\n', fontsize=font_s + 2)
    plt.xlabel('x', fontsize=font_s)
    plt.ylabel('y', fontsize=font_s)
    plt.legend()

    plt.grid()
    plt.show()


def plot_parzen_density_recovery(generated_data: np.array,
                                 parzen_f: Mapping,
                                 title: str,
                                 h: float = 1):
    
    font_s = 16
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(generated_data.min(), generated_data.max(), len(generated_data))
    parzen = parzen_f(x, generated_data, h)

    ax.plot(x, parzen, label='evaluated density')
    
    ax.scatter(generated_data, 
               np.zeros(len(generated_data)), 
               s=12, 
               marker='|',
               label='density projection')

    plt.title(f'Parzen Rozenblatt {title} Density\n', fontsize=font_s + 2)
    plt.xlabel('x', fontsize=font_s)
    plt.ylabel('density', fontsize=font_s)
    plt.legend()

    plt.grid()
    plt.show()


def plot_parzen_density_recovery(generated_data: np.array,
                                 parzen_f: Mapping,
                                 kernel: Mapping,
                                 title: str,
                                 h: float = 1,
                                 bins: float = 40):
    
    font_s = 16
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.linspace(generated_data.min(), generated_data.max(), len(generated_data))
    parzen = parzen_f(x, generated_data, kernel, h)

    ax.plot(x, parzen, label='evaluated density')
    
    ax.scatter(generated_data, 
               np.zeros(len(generated_data)), 
               s=20, 
               color='black',
               marker='|',
               label='density projection')
    
    ax.hist(generated_data, 
            bins=bins, 
            density=True, 
            alpha=0.2, 
            color='coral', 
            edgecolor='black',
            label='generated data density')
    
    sns.kdeplot(generated_data, 
                bw_method=.1, 
                color='coral',
                label='generated data density line')

    plt.title(f'{title} Density\n', fontsize=font_s + 2)
    plt.xlabel('x', fontsize=font_s)
    plt.ylabel('density', fontsize=font_s)
    plt.legend()
    
    plt.grid()
    plt.show()