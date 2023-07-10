"""
@author: Qizhi He (qzhe@umn.edu)
"""

import math
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy.linalg as la
plt.rcParams.update({'font.size': 20})

#%% plot Training data
def plotTrainData(train,color,path):
    PAD = 8
    e = train[:,0:3]
    s = train[:,3:6]
    
    # plot strain
    cm = plt.cm.get_cmap('jet')
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_subplot(121)
    ax.scatter(e[:,0], e[:,1],marker='o',s=40,c=color,cmap=cm)
    ax.tick_params(labelsize=12)
    ax.set_title('Training - Strain',fontsize=20,pad=PAD)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.tick_params(axis='both',which='major',pad = PAD)
    ax.set_xlabel('$E_{11}$',fontsize=20,labelpad=PAD)
    ax.set_ylabel('$E_{22}$',fontsize=20,labelpad=PAD)
    plt.grid()
    plt.tight_layout()
#    plt.savefig('./figures/Training_Strain.png', \
#                bbox_inches='tight', pad_inches = 0)

    # plot stess
    ax = fig.add_subplot(122)
    ax.scatter(s[:,0], s[:,1],marker='o',s=40,c=color,cmap=cm)
    ax.tick_params(labelsize=12)
    ax.set_title('Training - Stress',fontsize=20,pad=PAD)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.tick_params(axis='both',which='major',pad = PAD)
    ax.set_xlabel('$S_{11}$',fontsize=20,labelpad=PAD)
    ax.set_ylabel('$S_{22}$',fontsize=20,labelpad=PAD)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + './figures/Training data.png', \
                bbox_inches='tight', pad_inches = 0)
    return

#%% plot NN Output
def plotNNoutput(output,color,path):
    PAD = 8
    e = output[:,0:3]
    s = output[:,3:6]
    
    # plot strain
    cm = plt.cm.get_cmap('jet')
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_subplot(121)
    ax.scatter(e[:,0], e[:,1],marker='o',c=color,cmap=cm)
    ax.tick_params(labelsize=12)
    ax.set_title('AE Output - Strain',fontsize=20,pad=PAD)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.tick_params(axis='both',which='major',pad = PAD)
    ax.set_xlabel('$E_{11}^{NN}$',fontsize=20,labelpad=PAD)
    ax.set_ylabel('$E_{22}^{NN}$',fontsize=20,labelpad=PAD)
    plt.grid()
    plt.tight_layout()
#    plt.savefig('./figures/NN Output_Training_Strain.png', \
#                bbox_inches='tight', pad_inches = 0)

    # plot stess
    ax = fig.add_subplot(122)
    ax.scatter(s[:,0], s[:,1],marker='o',c=color,cmap=cm)
    ax.tick_params(labelsize=12)
    ax.set_title('AE Output - Stress',fontsize=20,pad=PAD)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.tick_params(axis='both',which='major',pad = PAD)
    ax.set_xlabel('$S_{11}^{NN}$',fontsize=20,labelpad=PAD)
    ax.set_ylabel('$S_{22}^{NN}$',fontsize=20,labelpad=PAD)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + './figures/AE Output_Training data.png', \
                bbox_inches='tight', pad_inches = 0)
    return


#%% plot Testing Stress
def plotTest(test,pred,path):
    PAD = 8
    
    fig = plt.figure(figsize=(11,5))
    ax = fig.add_subplot(121)
    ax.scatter(test[:,0], test[:,1],c='b',zorder=1)
    ax.scatter(pred[:,0], pred[:,1], facecolors='none',edgecolors='r',c='r',zorder=10)
    ax.tick_params(labelsize=12)
    ax.set_title('Testing - Strain', fontsize=20,pad=20)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.tick_params(axis='both',which='major',pad = PAD)
    ax.set_xlabel('$E_{11}$',fontsize=20,labelpad=PAD)
    ax.set_ylabel('$E_{22}$',fontsize=20,labelpad=PAD)
    plt.grid()
    plt.tight_layout()
    
    ax = fig.add_subplot(122)
    ax.scatter(test[:,3], test[:,4],c='b',zorder=1)
    ax.scatter(pred[:,3], pred[:,4], facecolors='none',edgecolors='r',c='r',zorder=10)
    ax.tick_params(labelsize=12)
    ax.set_title('Testing - Stress', fontsize=20,pad=20)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
    ax.tick_params(axis='both',which='major',pad = PAD)
    ax.set_xlabel('$S_{11}$',fontsize=20,labelpad=PAD)
    ax.set_ylabel('$S_{22}$',fontsize=20,labelpad=PAD)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + './figures/Comparison_Test.png', \
                bbox_inches='tight', pad_inches = 0)
    
    return

#%% plot embedding
def plotEmbedding(x,color,path):
    PAD = 8
        
    cm = plt.cm.get_cmap('jet')
    fig = plt.figure(figsize=(10,5))
    ax = fig.add_subplot(121)
    ax.scatter(x[:,0], x[:,1], marker='o',s=40,c=color,cmap=cm)
    ax.tick_params(labelsize=12)
    ax.set_title('Trained Embedding - 2D view',fontsize=20,pad=PAD)
    plt.tight_layout()
    
    if x.shape[1] > 2:
        ax = fig.add_subplot(122,projection='3d')
        ax.scatter(x[:,0], x[:,1], x[:,2], marker='o',s=40,c=color,cmap=cm)
        ax.view_init(15,-45)
    ax.tick_params(labelsize=12)
    ax.set_title('Trained Embedding - 3D view', fontsize=20,pad=20)
    ax.tick_params(axis='both',which='major',pad = PAD)
    plt.grid()
    plt.tight_layout()
    plt.savefig(path + './figures/trained_embedding.png', \
                bbox_inches='tight', pad_inches = 0)
    
    return