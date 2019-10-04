import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
# for overlaying images:
from matplotlib import offsetbox

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable


## Plotting functions ------------------------------------------------------

def plot2D(X=np.ndarray([]), label=np.array([]), 
           figsize=(10, 10), title=None, 
           col_map=plt.cm.Spectral, **kwargs):
    if len(label) > 0 and X.shape[0] != len(label):
        raise ValueError("Number of rows in X must equal length of label, if given.")
    ulabs = np.sort(np.unique(label))
    plt.figure(figsize=figsize)
    if isinstance(title, str):
        plt.title(title)
    if len(label) == 0:
        plt.scatter(X[:,0], X[:,1])        
    elif any([isinstance(lab, str) for lab in ulabs]) or len(ulabs) <= 10:
        for i, lab in enumerate(ulabs):
            if type(col_map) == type([]):
                plt.scatter(X[label==lab,0], X[label==lab,1], 
                        edgecolor='black', linewidth=0.1,
                        label=str(lab), c = col_map[i], **kwargs)  
            else:
                plt.scatter(X[label==lab,0], X[label==lab,1], 
                            edgecolor='black', linewidth=0.1,
                            label=str(lab), **kwargs)
        #plt.legend()
    else:
        plt.scatter(X[:,0], X[:,1],  
                    edgecolor='black', linewidth=0.1,
                    cmap=col_map, c=label, **kwargs)
        plt.colorbar(shrink = 0.8)
    #plt.axes().set_aspect('equal')
    return
    
def plot3D(X=np.ndarray([]), label=np.array([]), title=None, 
           figsize=(12, 10), phi = 20, theta = 60,
           col_map=plt.cm.Spectral, col_bar = True):
    if len(label) > 0 and X.shape[0] != len(label):
        raise ValueError("Number of rows in X must equal length of label, if given.")
    
    ulabs = np.unique(label)
    if any([isinstance(lab, str) for lab in ulabs]):
        label = [i for i, cat in enumerate(ulabs) for lab in label if lab == cat]
    
    fig = plt.figure(figsize=figsize)
    if isinstance(title, str):
        plt.suptitle(title)
    ax = fig.add_subplot(111, projection='3d')
    if len(label) == 0:
        ax.scatter(X[:, 0], X[:, 1],X[:, 2])   
    else:
        p = ax.scatter(X[:, 0], X[:, 1],X[:, 2], 
                       c=label, s=50, cmap=col_map,
                       edgecolor='black', linewidth=0.1)
        if col_bar:
            fig.colorbar(p, shrink = 0.7)
        
    max_range = np.array([X[:, 0].max() - X[:, 0].min(), 
                          X[:, 1].max() - X[:, 1].min(), 
                          X[:, 2].max() - X[:, 2].min()]).max() / 2.0
    mid_x = (X[:, 0].max() + X[:, 0].min()) * 0.5
    mid_y = (X[:, 1].max() + X[:, 1].min()) * 0.5
    mid_z = (X[:, 2].max() + X[:, 2].min()) * 0.5
    ax.set_xlim3d(mid_x - max_range, mid_x + max_range)
    ax.set_ylim3d(mid_y - max_range, mid_y + max_range)
    ax.set_zlim3d(mid_z - max_range, mid_z + max_range)
    ax.view_init(phi, theta)
    ax.set_aspect(1.0)
    return p


#----------------------------------------------------------------------
# Scale and visualize the embedding vectors

def plot2D_with_images(X, labels, images, title=None, figsize=(10, 8)):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=figsize)
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.tab10(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 16})

    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(X.shape[0]):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i], cmap=plt.cm.gray_r),
                X[i])
            ax.add_artist(imagebox)
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)

#----------------------------------------------------------------------
## Plot list of 2D embeddings:

def plot_embdeddings(X_lst, color, name_lst=None, title="",
                     figsize=(15, 8), fontsize = 14, color_map=None,
                     axis_equal = True, discrete_colors=False,  ncol = 5, **kwargs):
    if 'ncol' in kwargs.keys():
        ncol =  kwargs['ncol'] 
        del kwargs['ncol'] 
    nrow = np.ceil(len(X_lst) / ncol)
    if((len(color) != len(X_lst)) and (len(color) == X_lst[0].shape[0])):
        color = [list(color)] * len(X_lst)
    if color_map is None and not discrete_colors:
        color_map = plt.cm.Spectral
    elif color_map is None and discrete_colors:
        cmap = mpl.cm.get_cmap('Spectral')
        max_num_col = max([len(np.unique(c)) for c in color])
        color_map = [mpl.colors.rgb2hex(cmap(x)) 
                     for x in np.linspace(0, 1, max_num_col)]    
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=fontsize)
    for i, X in enumerate(X_lst):
        icol = color[i]
        ax = fig.add_subplot(nrow, ncol, 1 + i)
        if discrete_colors:
            for k, col in enumerate(np.unique(icol)):
                idx = (np.array(icol) == col)
                plt.scatter(X[idx, 0], X[idx,1], label=str(col), 
                            c=color_map[k], **kwargs)  
        else:    
            plt.scatter(X[:, 0], X[:, 1], c=icol, cmap=color_map, **kwargs)
        if name_lst is not None:
            name = name_lst[i]
            plt.title(name, fontsize=fontsize)
        if axis_equal:
            plt.axis('equal')
    return

    

# import plotly.plotly as py
# import plotly.graph_objs as go
def plotly_3D(x, y, z, label = None, title="", 
              size = 3, colors='Viridis'):
    try:
        __import__(module_name)
    except ImportError:
        return False
    else:
        if label is not None: 
            trace1 = plotly.graph_objs.Scatter3d(
                x=x, y=y, z=z, mode='markers',
                marker=dict(
                    size=size,
                    color=label,                # set color to an array/list of desired values
                    colorscale=colors,          # choose a colorscale
                    opacity=0.8
                )
            )
        else: 
            trace1 = plotly.graph_objs.Scatter3d(
                x=x,y=y,z=z,
                mode='markers',
                marker=dict(
                    color='rgb(127, 127, 127)',
                    size=size,
                    opacity=0.8,
                    line=dict(
                        color='rgb(204, 204, 204)',
                        width=0.5)        
                )
            )
        data = [trace1]
        layout = plotly.graph_objs.Layout(margin=dict(l=0, r=0, b=0, t=0))
        fig = plotly.graph_objs.Figure(data=data, layout=layout)
        return fig

# fig = plt.figure(figsize=(17, 6))
# fig.suptitle(r"bandwidth $\log(q_0)$ vs $\log(\sigma_{tsne}$)")
# ax = fig.add_subplot(1, 2, 1, projection='3d')
# p = ax.scatter(
#     X[:, 0], X[:, 1],X[:, 2], 
#     c=np.log(q0), s=50, cmap=plt.cm.RdBu, 
#     edgecolor='black', linewidth=0.1)
# fig.colorbar(p)
# ax = fig.add_subplot(1, 2, 2,  projection='3d')
# p = ax.scatter(
#     X[:, 0], X[:, 1],X[:, 2], 
#     c=np.log(sigma.ravel()), s=50, cmap=plt.cm.RdBu, 
#     edgecolor='black', linewidth=0.1)
# fig.colorbar(p)

# Do the following once only
# import colorlover as cl
# import matplotlib as mpl
# import plotly.plotly as py
# from matplotlib.colors import LogNorm
# from mpl_toolkits.mplot3d import Axes3D
# import plotly 
# plotly.tools.set_credentials_file(username='nlhuong', api_key='blablabla')
# fig = plotly_3D(X[:, 0], X[:, 1], X[:, 2], label = color, size = 4, 
#                 colors='Rainbow')
# py.iplot(fig, filename='Swiss Roll Uniform')