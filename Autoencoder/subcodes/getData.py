"""
@author: Xiaolong HE
"""

import numpy as np
import numpy.linalg as la
from scipy.io import loadmat
np.random.seed(1)

#%% Get Data from LinearElastic model
def getDataLinearElastic(nnode, plane_type = 1):
    nTr = nnode**3
    
    # Strain Data
    h = np.linspace(-5e-4, 5e-4, nnode, dtype='float32')
#   h = np.linspace(-1e-2, 1e-2, nnode, dtype='float32')
    h11, h22, h12 = np.meshgrid(h,h,h)
    e = np.concatenate((h11.reshape(1,nTr), h22.reshape(1,nTr), h12.reshape(1,nTr)), axis=0)
    
    # Add Noise to Strain Data
#    e = e + np.float32(r*np.random.rand(3,nTr))
#    e = e + np.float32(r*2e-2*np.random.rand(3,nTr))
    
    # Stress Data
#    E = 3e7    # psi, Young's modulus
#    v = 0.25    # Poisson's ratio
    E = 3e7    # MPa, Young's modulus
    v = 0.3    # Poisson's ratio
    
    if plane_type == 1 : # Plane Stress
        C = E/(1-v**2) * np.array([[1, v, 0], [v, 1, 0], [0, 0, (1-v)/2]], dtype='float32')
    elif plane_type == 2: # Plane Strain
        C = E/(1+v)/(1-2*v) * np.array([[1-v, v, 0], [v, 1-v, 0], [0, 0, (1-2*v)/2]], dtype='float32')
    
    s = np.dot(C,e)
    
    e = e.transpose()
    s = s.transpose()
    return e,s,nTr


#%% Get Data from HyperElastic model
#def getDataStVenant(nnode):
#    # Deformation gradient
#    F11 = np.linspace(5e-1, 1.1, nnode, dtype='float32')
#    F12 = np.linspace(-0.1, 0.8, nnode, dtype='float32')
#    F21 = np.linspace(-0.8, 0.1, nnode, dtype='float32')
#    F22 = np.linspace(5e-1, 1.1, nnode, dtype='float32')
#    nTr = len(F11) * len(F12) * len(F21) * len(F22)
#    F = np.zeros((3,3,nTr), dtype = 'float32')
#    c = -1
#    for i in range(nnode):
#        for j in range(nnode):
#            for k in range(nnode):
#                for l in range(nnode):
#                    c += 1
#                    F[:,:,c] = [[F11[i], F12[j], 0.0],[F21[k], F22[l], 0.0],[0.0, 0.0, 1.0]]
#    s = np.zeros((3,nTr), dtype = 'float32')
#    e = np.zeros((3,nTr), dtype = 'float32')
#    
#    # Elastic tensor
#    E = 3.0e4 # psi
#    v = 0.25
#    Cel = E/(1+v)/(1-2*v) * np.array([[1-v, v, 0], [v, 1-v, 0], [0, 0, (1-2*v)/2]], dtype='float32')    
#    
#    for i in range(nTr):
#        # Cauchy Green tensor
#        F1 = F[:,:,i]
#        C = np.dot(F1.transpose(),F1)
#        delta = np.eye(3)
#        E = 0.5 * (C - delta)
#        
#        # 2nd Piola-Kirchoff stress S
#        S = np.dot(Cel,E)
#        
#        # Voigt notation of stress and strain
#        s[:,i] = [S[0,0], S[1,1], S[0,1]]
#        e[:,i] = [E[0,0], E[1,1], E[0,1]]
#    
#    e = e.transpose()
#    s = s.transpose()
#    return e,s


def getDataStVenant(nnode):
    # Deformation gradient
    F11 = np.linspace(5e-1, 1.1, nnode, dtype='float32')
    F12 = np.linspace(-0.1, 0.8, nnode, dtype='float32')
    F21 = np.linspace(-0.8, 0.1, nnode, dtype='float32')
    F22 = np.linspace(5e-1, 1.1, nnode, dtype='float32')
    nTr = len(F11) * len(F12) * len(F21) * len(F22)
    F = np.zeros((3,3,nTr), dtype = 'float32')
    c = -1
    for i in range(nnode):
        for j in range(nnode):
            for k in range(nnode):
                for l in range(nnode):
                    c += 1
                    F[:,:,c] = [[F11[i], F12[j], 0.0],[F21[k], F22[l], 0.0],[0.0, 0.0, 1.0]]
    s = np.zeros((3,nTr), dtype = 'float32')
    e = np.zeros((3,nTr), dtype = 'float32')
    
    # Tensor Cs
    delta = np.eye(3)
    E = 3.0e4 # psi, Young's modulus
    v = 0.25 # Poisson ratio
#    lamb = v*E / (1+v) / (1-2*v)
#    mu = E / 2 / (1+v)
#    Cs = np.zeros((3,3,3,3),dtype='float32')    
#    for i in range(3):
#        for j in range(3):
#            for k in range(3):
#                for l in range(3):
#                    Cs[i,j,k,l] = Cs[i,j,k,l] + lamb * delta[i,j] * delta[k,l] \
#                        + mu * (delta[i,k] * delta[j,l] + delta[i,l] * delta[j,k])
#    
#    Cs_voigt = np.zeros((3,3),dtype='float32') 
#    Cs_voigt = [[Cs[1,1,1,1],Cs[1,1,2,2],Cs[1,1,1,2]],
#                [Cs[2,2,1,1],Cs[2,2,2,2],Cs[2,2,1,2]],
#                [Cs[1,2,1,1],Cs[1,2,2,2],Cs[1,2,1,2]]] 
              
    Cs_voigt = E/(1+v)/(1-2*v) * np.array([[1-v, v, 0], 
                                    [v, 1-v, 0],
                                    [0, 0, (1-2*v)/2]], dtype='float32')                
    for i in range(nTr):
        # Cauchy Green tensor
        F1 = F[:,:,i]
        C = np.dot(F1.transpose(),F1)
        
        # Strain in Voigt form
        E = 0.5 * (C - delta)
        e[:,i] = [E[0,0], E[1,1], 2*E[0,1]] 
        
        # Add Noise to Strain Data
#        e[:,i] = e[:,i] + np.float32(r*np.random.rand(1,3))
        
        # 2nd Piola-Kirchoff Stress in Voigt form
        s[:,i] = np.dot(Cs_voigt,e[:,i])
    
    e = e.transpose()
    s = s.transpose()
    return e,s,nTr

def getDataStVenant2(filename): # input StVenant data from files
    mat = loadmat(filename)
    e = mat['strain'].astype('float32')  # noiseless strain
    s = mat['stress'].astype('float32')  # noiseless stress
    
    en = mat['strain_noisy'].astype('float32') # noisy strain if noise is added
    sn = mat['stress_noisy'].astype('float32') # noisy stress if noise is added
    
    # en = mat['strain_orig'].astype('float32') # noisy strain if noise is added
    # sn = mat['stress_orig'].astype('float32') # noisy stress if noise is added
    
    nTr = e.shape[0]
    return e,s,en,sn,nTr


def getDataStVenant3(nnode):
    # Green Strain
    E11 = np.linspace(-15e-2, 15e-2, nnode, dtype='float32')
    E12 = np.linspace(-2e-2, 2e-2, nnode, dtype='float32')
    E22 = np.linspace(-2e-2, 2e-2, nnode, dtype='float32')
    npt = nnode**3
    Ex,Ey,Ez = np.meshgrid(E11,E12,E22)
    e = np.concatenate((Ex.reshape((1,npt)),Ey.reshape((1,npt)),Ez.reshape((1,npt))),axis=0)
    
    # Tensor Cs - Plane Strain
    E = 4.8e3 # psi, Young's modulus
    v = 0 # Poisson ratio
    Cs_voigt = E/(1+v)/(1-2*v) * np.array([[1-v, v, 0], 
                                    [v, 1-v, 0],
                                    [0, 0, (1-2*v)/2]], dtype='float32')      
    
    # 2nd PK Stress
    s = np.matmul(Cs_voigt,e)
    
    e = e.transpose()
    s = s.transpose()
    return e,s,npt

def getDataMooneyRivlin(nnode):
    # Material constants
    A10 = 2.0e3 # psi
    A01 = 1.75e3 # psi
    K = 1e8 # psi
    
    # Deformation gradient
    F11 = np.linspace(9e-1, 1.05, nnode, dtype='float32')
    F12 = np.linspace(-0.1, 0.4, nnode, dtype='float32')
    F21 = np.linspace(-0.4, 0.1, nnode, dtype='float32')
    F22 = np.linspace(9e-1, 1.05, nnode, dtype='float32')
    nTr = len(F11) * len(F12) * len(F21) * len(F22)
    F = np.zeros((3,3,nTr), dtype = 'float32')
    c = -1
    for i in range(nnode):
        for j in range(nnode):
            for k in range(nnode):
                for l in range(nnode):
                    c += 1
                    F[:,:,c] = [[F11[i], F12[j], 0.0],[F21[k], F22[l], 0.0],[0.0, 0.0, 1.0]]
    s = np.zeros((3,nTr), dtype = 'float32')
    e = np.zeros((3,nTr), dtype = 'float32')
    
    for i in range(nTr):
        # Cauchy Green tensor
        F1 = F[:,:,i]
        C = np.dot(F1.transpose(),F1)
        Cinv = np.linalg.inv(C)
        J = np.linalg.det(F1)
        I1 = np.trace(C)
        I2 = 0.5 * (I1**2 - np.trace(np.dot(C,C)))
        I3 = J*J
        delta = np.eye(3)
        E = 0.5 * (C - delta)
        
        # 2nd Piola-Kirchoff stress S
        S = 2 * A10 * I3**(-1/3) * (delta - 1/3*I1*Cinv) + \
            2 * A01 * I3**(-2/3) * (I1*delta - C - 2/3*I2*Cinv) + \
            K * (J - 1) * J * Cinv
        
        # Voigt notation of stress and strain
        s[:,i] = [S[0,0], S[1,1], S[0,1]]
        e[:,i] = [E[0,0], E[1,1], E[0,1]]
    
    e = e.transpose()
    s = s.transpose()
    return e,s

#%% Bio-tissue noiseless data
def getDataBio(filename,trainID,testID):
    data = np.loadtxt(filename)

    e_train = np.empty([0,2],dtype='float32')
    s_train = np.empty([0,2],dtype='float32')
    for i in trainID:
        e_train = np.append(e_train,data[data[:,0] == i,5:7],axis=0)
        s_train = np.append(s_train,data[data[:,0] == i,7:9],axis=0) 
    e_train = np.append(e_train,np.zeros((e_train.shape[0],1)),axis=1)
    s_train = np.append(s_train,np.zeros((s_train.shape[0],1)),axis=1)
    orig_data_train = np.append(e_train,s_train,axis=1).astype('float32')
    
    e_test = np.empty([0,2],dtype='float32')
    s_test = np.empty([0,2],dtype='float32')
    for i in testID:
        e_test = np.append(e_test,data[data[:,0] == i,5:7],axis=0)
        s_test = np.append(s_test,data[data[:,0] == i,7:9],axis=0)
    e_test = np.append(e_test,np.zeros((e_test.shape[0],1)),axis=1)
    s_test = np.append(s_test,np.zeros((s_test.shape[0],1)),axis=1)
    orig_data_test = np.append(e_test,s_test,axis=1).astype('float32')
        
    return orig_data_train, orig_data_test

#%% Bio-tissue niosy data
def getNoisyDataBio(filename,trainID,testID,noise_rand):
    data = np.loadtxt(filename)

    ### training data
    e_train = np.empty([0,3],dtype='float32')
    s_train = np.empty([0,3],dtype='float32')
    e_train_noisy = np.empty([0,3],dtype='float32')
    s_train_noisy = np.empty([0,3],dtype='float32')
    for i in trainID:
        e = data[data[:,0] == i,5:7]
        s = data[data[:,0] == i,7:9]
        nTr = e.shape[0]
        e = np.append(e,np.zeros((nTr,1)),axis=1)
        s = np.append(s,np.zeros((nTr,1)),axis=1)
    
        ### add uniform distributed noise
        e_noisy = np.zeros(np.shape(e),'float32')
        s_noisy = np.zeros(np.shape(s),'float32')
        e_h = []
        s_h = []
        for i in range(e.shape[0]-1):
            e_h.append(la.norm(e[i,:]-e[i+1,:]))
        for i in range(s.shape[0]-1):
            s_h.append(la.norm(s[i,:]-s[i+1,:]))
        e_hmax = max(e_h)
        s_hmax = max(s_h)
        sf_e = noise_rand * e_hmax
        sf_s = noise_rand * s_hmax
        for i in range(3):  
            e_noisy[:,i:i+1] = e[:,i:i+1] + np.float32(sf_e * np.random.rand(nTr,1))
            s_noisy[:,i:i+1] = s[:,i:i+1] + np.float32(sf_s * np.random.rand(nTr,1))
        ###
        e_train = np.append(e_train,e,axis=0)
        s_train = np.append(s_train,s,axis=0) 
        e_train_noisy = np.append(e_train_noisy,e_noisy,axis=0)
        s_train_noisy = np.append(s_train_noisy,s_noisy,axis=0) 
#    e_train = np.append(e_train,np.zeros((e_train.shape[0],1)),axis=1)
#    s_train = np.append(s_train,np.zeros((s_train.shape[0],1)),axis=1)
#    e_train_noisy = np.append(e_train_noisy,np.zeros((e_train_noisy.shape[0],1)),axis=1)
#    s_train_noisy = np.append(s_train_noisy,np.zeros((s_train_noisy.shape[0],1)),axis=1)
    orig_data_train = np.append(e_train,s_train,axis=1).astype('float32')
    noisy_data_train = np.append(e_train_noisy,s_train_noisy,axis=1).astype('float32')
    
    # testing data
    e_test = np.empty([0,2],dtype='float32')
    s_test = np.empty([0,2],dtype='float32')
    for i in testID:
        e_test = np.append(e_test,data[data[:,0] == i,5:7],axis=0)
        s_test = np.append(s_test,data[data[:,0] == i,7:9],axis=0)
    e_test = np.append(e_test,np.zeros((e_test.shape[0],1)),axis=1)
    s_test = np.append(s_test,np.zeros((s_test.shape[0],1)),axis=1)
    orig_data_test = np.append(e_test,s_test,axis=1).astype('float32')
        
    return orig_data_train, noisy_data_train, orig_data_test