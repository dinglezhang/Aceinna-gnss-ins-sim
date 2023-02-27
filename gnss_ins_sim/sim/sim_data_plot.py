# -*- coding: utf-8 -*-
# Fielname = sim_data_plot.py

"""
Simulation data plot.
Created on 2020-07-24
@author: dongxiaoguang
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from mpl_toolkits.mplot3d import Axes3D
from . import sim_data

def plot(x, y, key, plot3d, mpl_opt=''):
    '''
    Plot x and y.
    Args:
        x: a Sim_data object.
        y: a Sim_data object.
        key: a tuple or list of keys corresponding to y.data.
        plot3d:
            1: 3D plot
            2: 3D plot projected on xy, xz and yz,
            3: 3D plot multiple data (dict type) into one figure
            otherwise: 2D plot
        mpl_opt: strings to specify matplotlib properties.
    '''
    x_data = None
    if isinstance(x.data, dict):
        if not x.data:  # x.data could be an empty dict
            x_data = None
        else:
            # choose first data
            for x_key in x.data.keys():
                x_data = x.data[x_key]
                break
    else:
        x_data = x.data

    if isinstance(y.data, dict) and (plot3d != 3):
        if key == []:
            key = y.data.keys()
        for y_key in key:
            y_data = y.data[y_key]
            plot_one_figure(x, x_data, y, y_data, plot3d, mpl_opt, ('_' + str(y_key)))
    else:
        plot_one_figure(x, x_data, y, y.data, plot3d, mpl_opt)

def plot_one_figure(x, x_data, y, y_data, plot3d, mpl_opt='', title_suffix=''):
    '''
    self.data is a numpy.array
    Args:
        x: a Sim_data object.
        x_data: specific x axis data, gotten from x
        y: a Sim_data object.
        y_data: specific y axis data, gotten from y
        title_suffix: plot title suffix
        plot3d: see plot()
        mpl_opt: see plot()
    '''
    # y axis
    # unit conversion
    y_data = sim_data.convert_unit(y_data, y.units, y.output_units)
    # plot
    if plot3d == 1 or plot3d == 3:
        plot_3d_figure(y_data,\
                       title=y.name + title_suffix,\
                       grid=y.grid,\
                       legend=y.legend,\
                       mpl_opt=mpl_opt)
    elif plot3d == 2:
        plot_3d_proj_figure(y_data,\
                            title=y.name + title_suffix,\
                            grid=y.grid,\
                            legend=y.legend,\
                            mpl_opt=mpl_opt)
    else:
        plot_2d_figure(x_data, y_data,\
                       logx=y.logx, logy=y.logy,\
                       xlabel=x.name + ' (' + x.output_units[0] + ')',\
                       ylabel=y.name + ' (' + str(y.output_units) + ')',\
                       title=y.name + title_suffix,\
                       grid=y.grid,\
                       legend=y.legend,\
                       mpl_opt=mpl_opt)

def plot_2d_figure(x_data, y_data,
                   logx=False, logy=False,\
                   xlabel=None, ylabel=None,\
                   title='Figure', grid='on', legend=None,\
                   mpl_opt=''):
    '''
    Create a figure and plot x/y in this figure.
    Args:
        x_data: x axis data, np.array of size (n,) or (n,1)
        y_data: y axis data, np.array of size (n,m)
        title: figure title
        xlabel: x axis label
        ylabel: y axis label
        gird: if this is not 'off', it will be changed to 'on'
        legend: tuple or list of strings of length m.
    '''
    # create figure and axis
    fig = plt.figure(title)
    axis = fig.add_subplot(111)
    lines = []
    # if not x data, generate default x data
    if x_data is None:
        x_data = np.array(range(y_data.shape[0]))
    try:
        dim = y_data.ndim
        if dim == 1:
            if logx and logy:   # loglog
                line, = axis.loglog(x_data, y_data, mpl_opt)
            elif logx:          # semilogx
                line, = axis.semilogx(x_data, y_data, mpl_opt)
            elif logy:          # semilogy
                line, = axis.semilogy(x_data, y_data, mpl_opt)
            else:               # plot
                line, = axis.plot(x_data, y_data, mpl_opt)
            lines.append(line)
        elif dim == 2:
            for i in range(0, y_data.shape[1]):
                if logx and logy:   # loglog
                    line, = axis.loglog(x_data, y_data[:, i], mpl_opt)
                elif logx:          # semilogx
                    line, = axis.semilogx(x_data, y_data[:, i], mpl_opt)
                elif logy:          # semilogy
                    line, = axis.semilogy(x_data, y_data[:, i], mpl_opt)
                else:               # plot
                    line, = axis.plot(x_data, y_data[:, i], mpl_opt)
                lines.append(line)
        else:
            raise ValueError
    except:
        print('x-axis data len: ', x_data.shape)
        print('y-axis data shape: ', y_data.shape)
        raise ValueError('Check input data y.')
    # label
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    # legend
    if legend is not None:
        plt.legend(lines, legend)
    # grid
    if grid.lower() != 'off':
        plt.grid()

def plot_3d_figure(y_data, title='Figure', grid='on', legend=None, mpl_opt=''):
    '''
    Create a figure and plot 3d trajectory in this figure.
    Args:
        y_data: y axis data, it may be:
            np.array of size (n,3)
            dict with value as np.array of size (n,3), that means to plot multiple 3d data into one figure
        title: figure title
        gird: if this is not 'off', it will be changed to 'on'
        legend: tuple or list of strings of length 3.
    '''
    # create figure and axis
    fig = plt.figure(title)
    axis = fig.add_subplot(111, projection='3d', aspect='auto')

    show_plot_legend = True
    if not isinstance(y_data, dict): # chnage y to temporary dict if it is not dict
        y_data = {title: y_data}
        show_plot_legend = False

    try:
        for key in y_data.keys():
            cur_y_data = y_data[key]
            dim = cur_y_data.ndim
            if dim == 2:    # cur_y must be an numpy array of size (n,3), dim=2
                if cur_y_data.shape[1] != 3:
                    raise ValueError
                else:
                    axis.plot(cur_y_data[:, 0], cur_y_data[:, 1], cur_y_data[:, 2], mpl_opt, label=key)
            else:
                raise ValueError
    except:
        print(cur_y_data.shape)
        raise ValueError('Check input data y.')

    # label
    if isinstance(legend, (tuple, list)):
        n = len(legend)
        if n != 3:
            legend = ['x', 'y', 'z']
    else:
        legend = ['x', 'y', 'z']
    axis.set_xlabel(legend[0])
    axis.set_ylabel(legend[1])
    axis.set_zlabel(legend[2])

    if show_plot_legend:
        plt.legend()

    if grid.lower() != 'off':
        plt.grid()

def plot_3d_proj_figure(y_data, title='Figure', grid='on', legend=None, mpl_opt=''):
    '''
    Create a figure and plot 3d projection trajectory in this figure.
    Args:
        y_data: y axis data, np.array of size (n,3)
        title: figure title
        gird: if this is not 'off', it will be changed to 'on'
        legend: tuple or list of strings of length 3.
    '''
    # plot data
    try:
        dim = y_data.ndim
        if dim == 2:    # y must be an numpy array of size (n,3), dim=2
            if y_data.shape[1] != 3:
                raise ValueError
            else:
                # check label
                if isinstance(legend, (tuple, list)):
                    n = len(legend)
                    if n != 3:
                        legend = ['x', 'y', 'z']
                else:
                    legend = ['x', 'y', 'z']
                # check grid
                show_grid = False
                if grid.lower() != 'off':
                    show_grid = True
                # create figure and axis
                # xy
                fig = plt.figure(title)
                axis = fig.add_subplot(131, aspect='equal')
                axis.plot(y_data[:, 0], y_data[:, 1], mpl_opt)
                axis.set_xlabel(legend[0])
                axis.set_ylabel(legend[1])
                axis.grid(show_grid)
                # xz
                axis = fig.add_subplot(132, aspect='equal')
                axis.plot(y_data[:, 0], y_data[:, 2], mpl_opt)
                axis.set_xlabel(legend[0])
                axis.set_ylabel(legend[2])
                axis.grid(show_grid)
                # yz
                axis = fig.add_subplot(133, aspect='equal')
                axis.plot(y_data[:, 1], y_data[:, 2], mpl_opt)
                axis.set_xlabel(legend[1])
                axis.set_ylabel(legend[2])
                axis.grid(show_grid)
        else:
            raise ValueError
    except:
        print(y_data.shape)
        raise ValueError('Check input data y.')

def show_plot():
    '''
    Show all plots
    '''
    plt.show()
