#Mie Python Test file
#This code takes input for type of dust and uses genereated arrays of wavelength, and density
#and pulls from a file the Size distribution for a sample. then it finds the
# weighted average of the MAE. in a n,k plane of the complex refractive index.
#(real, imaginary) adapted from examples from author of MiePython package
#5/3/2021
#Emily Faber

import miepython
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from numpy import ma
from matplotlib import ticker, cm

#range for plots
#Can't start with zero, or miePython.mie will divide by 0 in calculations

#for hematite
type = int(input('Hematite = 1, Dust = 2, Goethite = 3: '))
if type == 1:
    ref_n = np.linspace(1,3.5, num = 200) #hematite
    ref_k = np.linspace(.000001,.8, num = 200) #hematite
    rho = 5.24
#For Dust:
if type == 2:
    ref_n = np.linspace(1.25,1.6, num = 200) #dust
    ref_k = np.linspace(.000001,.004, num = 200) #dust
    rho = 2.4
#For goethite
if type == 3:
    ref_n = np.linspace(1,3.5, num = 200) #goethite
    ref_k = np.linspace(.000001,.8, num = 200) #goethite
    rho = 3.5


#import size distribution - this file originally came from Dr. Adriana Rocha Lima
#First column has the radius of the particles in micrometers, the second column is the number of particles per size bin
df = pd.read_csv('Dist_HMT01F-Hematite_Area_PhotoImpact_forEmily.txt', delimiter = '\t')

#change the density by X%:
# rho = rho*0.90
#
# change the size distribution by X%:
# df['Size'] = df['Size']*0.90

''' un-comment this to show the size distribution from the given file
# plots the size distribution of the particles
plt.plot(df['Size'],df['Number'])
plt.title('Size distribution')
plt.ylabel('Number of particles')
plt.xlabel('Radius in microns')
plt.show()
'''

#wavelength to look at the particle at
wavelength = float(input("Wavelength in microns: ")) #.500 # microns
#Used to find final MAE and b_abs
total_MAE = [[0]*len(ref_n)]*len(ref_k)
total_MAE = np.array(total_MAE)
b_abs = [[0]*len(ref_n)]*len(ref_k)
b_abs = np.array(b_abs)
total_mass = 0
#for each particle size
for i in range(0, len(df['Size'])):
    #size is Radius
    #lists to store Qabs for each n,k combination
    my_qabs = []
    #size parameter using diameter
    x = (np.pi*2*df['Size'][i])/wavelength
    #calculate Qabs for each n,k combination using the MiePython function
    for n in ref_n:
        for k in ref_k:
            m = n-1.0j*k
            qext, qsca, qback, g = miepython.mie(m,x)
            my_qabs.append(qext - qsca)
    #convert to numpy array
    my_qabs = np.array(my_qabs)
    #convert to 2D array to make meshgrid
    my_qabs = my_qabs.reshape(len(ref_n),len(ref_k))
    '''https://stackoverflow.com/questions/8421337/rotating-a-two-dimensional-array-in-python'''
    my_qabs = list(reversed(list(zip(*my_qabs))))
    my_qabs = np.flipud(my_qabs)
    b_abs = b_abs + df['Number'][i]*(my_qabs)*np.pi*((df['Size'][i]/(10**6))**2)#m^2
    total_mass = total_mass + df['Number'][i]*(rho*(10**6))*(4/3)*np.pi*((df['Size'][i]/(10**6))**3)#grams
    #                                             g/m^3                       m^3
total_MAE = b_abs/total_mass
N,K = np.meshgrid(ref_n,ref_k)
fig,ax = plt.subplots()
# total_MAE = np.round(total_MAE,2)
# levels = np.linspace(0,np.max(np.round(total_MAE,2)),8)
levels = np.linspace(0,np.max(total_MAE),20)
# levels = np.round(levels, 2)
# total_MAE = np.round(total_MAE,2)
# cs = ax.contourf(N,K,total_MAE, levels = levels, fmt = '{:.2f}')
cs = ax.contourf(N,K,total_MAE, levels = levels)
# plt.clabel(cs, levels, fmt = '{:.2f}')
plt.title('MAE for Size Distribution at ' + str(wavelength) + ' micrometers ' + ' [$m^2$/g]')
plt.ylabel('k - imaginary')
# plt.yscale('log') uncomment this to see more 'fine scale' features
plt.xlabel('n - real')
fig.colorbar(cs, label = 'Total Mass Absorption Efficiency')
plt.show()
# plt.savefig('MAE_SizeDist'+str(wavelength)+'micrometers'+str(rho)+'density'+'.png', dpi = 200)
