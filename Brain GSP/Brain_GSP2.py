# -*- coding: utf-8 -*-
"""
Created on Thu Dec 10 11:02:32 2020

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
graph = pg.graphs.Graph(connectivity, gtype='human connectome',
                        lap_type='normalized')
graph.compute_fourier_basis()
graph.set_coordinates(kind='spring')
coordinates = sio.loadmat('Glasser360_2mm_codebook.mat')['codeBook']
graph.coords = np.stack([coordinates[0, 0][-2][0][region][:, 0]
                         for region in range(n_regions)])

n_eigenvectors = 5
graph.coords = graph.coords[:, :2]
signals = sio.loadmat('RS_TCs.mat')['RS_TCs']
assert signals.shape[0] == n_regions
THRESHOLD = 30

graph.compute_fourier_basis()

signals_fourier = graph.gft(signals)

# mean = np.mean(np.abs(signals_fourier), axis=1)
norm = np.linalg.norm(signals_fourier, axis=1)
var = np.var(signals_fourier, axis=1)

# Remove DC value (constant signal).
norm = norm[1:]
var = var[1:]

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(graph.e[1:], norm)
axes[1].loglog(graph.e[1:], norm)

for ax in axes:
    ax.fill_between(graph.e[1:], norm-3*var, norm+3*var, alpha=0.5)
#    ax.vlines(graph.e[THRESHOLD], norm.min(), norm.max())
    ax.add_patch(mpl.patches.Rectangle((0, norm.min()), graph.e[THRESHOLD],
                                       norm.max(), color='b', alpha=0.1))
    ax.add_patch(mpl.patches.Rectangle((graph.e[THRESHOLD], norm.min()),
                                       graph.e[-1]-graph.e[THRESHOLD],
                                       norm.max(), color='r', alpha=0.1))
    ax.set_ylim(0.8*norm.min(), 1.2*norm.max())
    ax.set_xlabel(r'eigenvalue $\lambda$')

U_low = np.zeros_like(graph.U)
U_high = np.zeros_like(graph.U)

U_low[:, :THRESHOLD] = graph.U[:, :THRESHOLD]
U_high[:, THRESHOLD:] = graph.U[:, THRESHOLD:]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
ax1.imshow(U_low)
ax2.imshow(U_high)

signals_aligned = U_low @ U_low.T @ signals
signals_liberal = U_high @ U_high.T @ signals

signals_aligned = graph.igft(signals_fourier[:, :THRESHOLD])
signals_liberal = graph.igft(signals_fourier[:, THRESHOLD:])


def g_low(eigenvalue):
    return eigenvalue < graph.e[THRESHOLD]


def g_high(eigenvalue):
    return eigenvalue >= graph.e[THRESHOLD]


g = pg.filters.Filter(graph, [g_low, g_high])
g.plot(show_sum=False)

signals_filtered = g.filter(signals, method='exact')
signals_aligned = signals_filtered[..., 0]
signals_liberal = signals_filtered[..., 1]

fig, axes = plt.subplots(1, 3, figsize=(20, 4))
axes[0].imshow(signals)
axes[1].imshow(signals_aligned)
axes[2].imshow(signals_liberal)
axes[0].set_title('all')
axes[1].set_title('aligned')
axes[2].set_title('liberal')
for ax in axes:
    ax.set_xlabel('time')
    ax.set_ylabel('region')

norm_aligned = np.linalg.norm(signals_aligned, axis=1)
norm_liberal = np.linalg.norm(signals_liberal, axis=1)
var_aligned = np.var(signals_aligned, axis=1)
var_liberal = np.var(signals_liberal, axis=1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
graph.plot_signal(norm_aligned, ax=ax1)
graph.plot_signal(norm_liberal, ax=ax2)
ax1.set_title('aligned')
ax2.set_title('liberal')
ax1.set_aspect('equal', adjustable='box')
ax2.set_aspect('equal', adjustable='box')
ax1.axis('off')
ax2.axis('off')
