# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 09:36:16 2020

@author: mahdi
"""

from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.animation
import matplotlib.pyplot as plt
from scipy import io as sio
import pygsp as pg
import IPython.display as ipd

mpl.rcParams['animation.embed_limit'] = 2**129
connectivity = sio.loadmat('SC.mat')['SC']
n_regions, n_regions = connectivity.shape
print(f'{n_regions} brain regions')

# fig, ax = plt.subplots(figsize=(16, 9))
# ax.spy(connectivity)

fig, ax = plt.subplots(figsize=(16, 9))
im = ax.imshow(np.log(connectivity, out=np.zeros_like(connectivity),
               where=(connectivity != 0)))
plt.colorbar(im)
ax.set_title('Connectivity matrix of the brain')

# fig, ax = plt.subplots(figsize=(16, 9))
# ax.hist(connectivity.flatten(), bins=100, log=True)
# ax.show()


graph = pg.graphs.Graph(connectivity, gtype='human connectome',
                        lap_type='normalized')

print(f'{graph.N:_} nodes')
print(f'{graph.Ne:_} edges')
print(f'connected: {graph.is_connected()}')
print(f'directed: {graph.is_directed()}')

fig, ax = plt.subplots(figsize=(16, 9))
ax.hist(graph.dw, bins=100)

graph.compute_fourier_basis()
fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(graph.e)
ax.set_title('eigenvalues of the connectivity matrix')

graph.set_coordinates(kind='spring')
graph.plot()


coordinates = sio.loadmat('Glasser360_2mm_codebook.mat')['codeBook']
graph.coords = np.stack([coordinates[0, 0][-2][0][region][:, 0]
                         for region in range(n_regions)])

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1, projection='3d')
graph.plot(ax=ax)
ax.set_aspect('auto', adjustable='box')

n_eigenvectors = 5

fig = plt.figure(figsize=(20, 5))
for i in range(n_eigenvectors):
    ax = fig.add_subplot(1, n_eigenvectors, i+1, projection='3d')
    ax.set_aspect('auto', adjustable='box')
    graph.plot_signal(graph.U[:, i], ax=ax, colorbar=False)
    ax.set_title(f'eigenvector {i}')
    ax.axis('off')

graph.coords = graph.coords[:, :2]

fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, ax in enumerate(axes):
    graph.plot_signal(graph.U[:, i], ax=ax, colorbar=True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'eigenvector {i}')
    ax.axis('off')

signals = sio.loadmat('RS_TCs.mat')['RS_TCs']
assert signals.shape[0] == n_regions

fig, ax = plt.subplots(figsize=(16, 9))
ax.plot(signals.T)
ax.set_title('fMRI signal across different regions of the brain')

# plt.plot(signals.mean(axis=0));

START = 200
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
for i, ax in enumerate(axes):
    graph.plot_signal(signals[:, START+i], ax=ax, colorbar=False)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title(f'time {START+i}')
    ax.axis('off')

plt.rc('animation', embed_limit=100)
fig, ax = plt.subplots(figsize=(5, 5))
ax.set_aspect('equal', adjustable='box')
ax.axis('off')
plt.close(fig)
sc = ax.scatter(graph.coords[:, 0], graph.coords[:, 1])
cmap = plt.cm.get_cmap()


def animate(i):
    sc.set_color(cmap(signals[:, i]))
    ax.set_title(f'time {i}')
    return (sc,)


fig = plt.figure()
animation = mpl.animation.FuncAnimation(fig, animate, blit=True,
                                        frames=signals.shape[1], interval=20)
plt.show()
ipd.HTML(animation.to_jshtml())
