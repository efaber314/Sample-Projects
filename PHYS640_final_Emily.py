#640 final project
#Emily Faber

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import scipy.integrate
import netCDF4
import pandas as pd
from scipy.interpolate import CubicSpline
from scipy.fftpack import fft

#π𝑅^2∙𝐹∙(1−𝛼)=4π𝑅^2∙𝜀𝜎𝑇^4
"""
Problem 1a uses the above equation to show the temperature of the earth's surface
if it did not have an atmosphere.
"""
def problem1a():
    a = 0.293 #planetary albedo
    F0 = 1378 #W/m^2
    epsilon = 1 #surface emissivity
    R = 6.3781*10**6 # radius of the Earth in meters
    sigma = 5.76*10**(-8) #W/m^2k^4 stephan boltzman constant
    #solve for Temperature of planet
    top = F0*(1-a)
    bottom = 4*epsilon*sigma
    temp = top/bottom
    T0 = temp**(1/4)

    return np.round(T0,3)
"""
problem 1b is a model that calculates the equilibrium surface temperature and
atmosphereic temperature profile for a N-layer atmosphere. (N>2)
"""

def problem1b(N, emissivity):
    surfaceEmissivity = 1.0
    a = 0.293 #albedo of earth
    F0 = 1378 #W/m^2
    #epsilon = 1 #emissivity of ground
    #R = 6.371*10**6 #m
    sigma = 5.67*10**(-8) #W/m^2k^4 stephan boltzman constant
    Incoming = F0/4 #incoming solar flux

    #matrix representing the values of the system of equations
    b = np.ones(N+1)
    #The surface value is a scalar representing the flux, rather than 0
    b[0] = -((1-a)*Incoming)
    #matrix of coeffiecients of the temperatures of the layers
    M = np.zeros((N+1,N+1))
    #fill the matrix with the proper coeffiecients
    for j in range(0,N+1):
        for i in range(0,N+1):
            if j<i:
                if j == 0:
                    M[j][i] = sigma*((1-emissivity)**(i-1))*emissivity*surfaceEmissivity
                else:
                    M[j][i] = sigma*((1-emissivity)**(i-1))*emissivity**2
            if j == i:
                if j == 0 and i == 0:
                    M[j][i] = -(surfaceEmissivity*sigma)
                else:
                    M[j][i] = -(2*emissivity*sigma)
            if j>i:
                if i == 0:
                    M[j][i] = sigma*((1-emissivity)**(j-1))*emissivity*surfaceEmissivity
                else:
                    M[j][i] = sigma*((1-emissivity)**(j-1))*emissivity**2
    T = np.linalg.solve(M,b) #solve using linear algebra
    T = np.abs(T)
    #remeber these are coeffiecients of T^4, so we must take the 4th root to get
    #the equilibrium temperature of each layer
    T = T**(1/4)

    return T

""" Using problem1b, this plots useful things"""
def problem1c():
    N = 10
    e = .5
    T = problem1b(N,e)
    fig,ax = plt.subplots()
    #plots vertical temperature profile for N = 10 and epsilon of the atmosphereic Layers
    # is 0.5
    ax.plot(T,np.arange(0,N+1), label = "N = 10, emissivity = .5")
    plt.legend()
    ax.set_xlim(300, 0)
    plt.xlabel('Temperature of layer [k]')
    plt.ylabel('Layer of system')
    plt.title('Temperature Profile of the Given Atmosphere')
    plt.savefig("PHYS640_final_P1c_fig1_Emily.png", dpi = 600)
    plt.close()

    surfaceT = []
    e = np.arange(.05,1,.05)
    for i in e:
        mySurfaceProfile = problem1b(N,i)
        mySurfaceTemp = mySurfaceProfile[0]
        surfaceT.append(mySurfaceTemp)
    #plots the surface temperature as a function of epsilon (ranging from 0 to 1
    # in steps of .05) for N = 10
    plt.plot(e,surfaceT, label = "N = 10")
    plt.legend()
    plt.xlabel('Emissivity')
    plt.ylabel('Surface Temperature [k]')
    plt.title('Surface Temerature as Atmosphere Emissivity Increases')
    plt.savefig("PHYS640_final_P1c_fig2_Emily.png", dpi = 600)
    plt.close()

    N = np.arange(3,20,1)
    e = .5
    surfaceT1 = []
    for n in N:
        mySurfaceProfile = problem1b(n,e)
        mySurfaceTemp = mySurfaceProfile[0]
        surfaceT1.append(mySurfaceTemp)
    #plots surface temperature as a function of N (Ranging from 3 to 20) for
    #epsilon of 0.5
    plt.plot(N,surfaceT1, label = "Emissivity = .5")
    plt.legend()
    plt.xlim((0,22))
    plt.xlabel('Number of Layers')
    plt.ylabel('Surface Temperature [k]')
    plt.title('Surface Temperature as the Number of Layers Increase')
    plt.savefig("PHYS640_final_P1c_fig3_Emily.png", dpi = 600)
    plt.close()

    return

""" This additional function fits the data from problem 1d """
def fitFunction(x,N, e):
    newX = []
    for elm in x:
        #This is a modified sigmoid function
        newX.append((elm + e/(e + np.exp(-(elm)))))
    return newX

''' Uses nonlinear regretion to fit a function of epsilon given N. Gives root
mean square error of regression estimates'''
def problem1d(N):
    #range of emissivity for same number of layers
    e = np.arange(.01,1,.01)
    n = [N]*len(e)
    surfaceTemp = []
    #find surface temperature for all the required emissivities
    for E in e:
        myTempProfile = problem1b(N,E)
        mySurfaceTemp = myTempProfile[0]
        surfaceTemp.append(mySurfaceTemp)
    #nonlinear curve fitting using the fitFunction
    optParam, covar = curve_fit(fitFunction, surfaceTemp, e, method = 'trf')

    #gets fit line
    XFit = np.arange(250,349,1)
    yFit = fitFunction(surfaceTemp,optParam[0],optParam[1])
    #calculate RMSE
    surfaceTemp = np.array(surfaceTemp)
    rmse = np.sqrt(((yFit - surfaceTemp) ** 2.0).mean())
    #plot data and fit line
    plt.figure()
    plt.plot(e, surfaceTemp, 'o', label='Data')
    plt.plot(e, yFit, c = 'r',linestyle = 'dotted', label='fit')
    s = "rmse = " + str(rmse)
    plt.text(.2, 300, s = s)
    plt.legend()
    plt.xlabel("Emissivity")
    plt.ylabel("Surface Temperature [k]")
    plt.title("Surface Temperature as a Function of Emissivity")
    plt.savefig("PHYS640_final_P1d_Fig1_Emily.png", dpi = 600)
    plt.close()

    return

''' Statistical analysis of the global dataset provided'''
def problem2a():

    data = netCDF4.Dataset('air.mon.mean.nc')
    mylat = data['lat'][20]
    mylong = data['lon'][25]
    print('Chosen latitude and longitude:',mylat, ",", mylong)

    #get data into usable format
    jan,feb,mar,apr,may,jun,jul,aug,sep,oct,nov,dec = ([] for i in range(12))

    for i in range(0,865):
        if i % 12 == 0:
            jan.append(data['air'][i][20][25])
        elif i % 12 == 1:
            feb.append(data['air'][i][20][25])
        elif i % 12 == 2:
            mar.append(data['air'][i][20][25])
        elif i % 12 == 3:
            apr.append(data['air'][i][20][25])
        elif i % 12 == 4:
            may.append(data['air'][i][20][25])
        elif i % 12 == 5:
            jun.append(data['air'][i][20][25])
        elif i % 12 == 6:
            jul.append(data['air'][i][20][25])
        elif i % 12 == 7:
            aug.append(data['air'][i][20][25])
        elif i % 12 == 8:
            sep.append(data['air'][i][20][25])
        elif i % 12 == 9:
            oct.append(data['air'][i][20][25])
        elif i % 12 == 10:
            nov.append(data['air'][i][20][25])
        elif i % 12 == 11:
            dec.append(data['air'][i][20][25])

    averages = []
    averages.append(np.mean(jan))
    averages.append(np.mean(feb))
    averages.append(np.mean(mar))
    averages.append(np.mean(apr))
    averages.append(np.mean(may))
    averages.append(np.mean(jun))
    averages.append(np.mean(jul))
    averages.append(np.mean(aug))
    averages.append(np.mean(sep))
    averages.append(np.mean(oct))
    averages.append(np.mean(nov))
    averages.append(np.mean(dec))

    #to make the calculations for the first and last two weeks of the year I
    #wrap the data back around.
    averages = np.insert(averages,0,averages[-1],axis = 0)
    averages = np.insert(averages,-1,averages[1],axis = 0)

    everyday = np.arange(1,365)
    #assuming that the average temperature occurs on the 15th of each month
    dataDays = [0,15,46,75,106,136,167,197,228,259,289,320,350,365] #day out of the year that is the 15th of each month (http://mistupid.com/calendar/dayofyear.htm)
    #extrapolate data for each day of the year
    everyday_temp = CubicSpline(dataDays,averages)
    plt.plot(dataDays,averages,'o',label = 'Data')
    plt.plot(everyday,everyday_temp(everyday), label = 'Interpolated Data')
    plt.legend()
    plt.xlabel("Day of the Year")
    plt.ylabel("Air Temperature [C]")
    plt.title("Interpolated Temperature Estimate Throughout the Year")
    plt.savefig("PHYS640_final_P2a_Emily.png",dpi = 600)
    plt.close()

    return
'''Problem2b calculates the fourier power spectrum of the monthly mean temperatures
of two locations. one in the mid latitudes and one in the equitorial region the
scale of the most prominent feature is shown and discussed in associated PDF '''
def problem2b():

    data = netCDF4.Dataset('air.mon.mean.nc')
    midlat = data['lat'][20]
    midlong = data['lon'][25]
    print('Chosen mid-latitude and mid-longitude:',midlat, ",", midlong)
    equlat = data['lat'][36]
    equlong = data['lon'][25]
    print("Chosen equitorial latitude and longitude:", equlat,equlong)
    mymonthsMIDLAT = []
    #want first 512 months
    N = 512
    for i in range(0,N):
        mymonthsMIDLAT.append(data['air'][i][20][25])
    mymonthsEQUI = []
    for j in range(0,N):
        mymonthsEQUI.append(data['air'][j][36][25])

    months = np.arange(1,N+1)
    plt.figure()
    plt.plot(months, mymonthsMIDLAT, label = 'MIDLAT')
    plt.plot(months, mymonthsEQUI, label = 'EQUITORIAL')
    plt.xlabel('Sequential Months [Jan 1948 - Aug 1990]')
    plt.ylabel('Temperature of Each Month [C]')
    plt.title('Temperature Profiles of 512 Sequential Months')
    plt.legend()
    plt.savefig("PHYS640_final_P2b_fig1_Emily.png", dpi = 600)
    plt.close()

    #get power spectrum
    yMIDLAT = fft(mymonthsMIDLAT)
    yEQUI = fft(mymonthsEQUI)
    e1 = np.abs(yMIDLAT)**2
    e2 = np.abs(yEQUI)**2
    NHalf = int(N/2)
    e1 = e1[0:NHalf]
    e2 = e2[0:NHalf]
    k1 = np.arange(0,len(e1))
    k2 = np.arange(0,len(e2))

    #plots fourier power spectrum
    plt.figure()
    plt.yscale('log')
    plt.xscale('log')
    plt.plot(k1,e1, label = 'Midlatitude (40, 62.5)')
    plt.plot(k2,e2, label = 'Equitorial (0, 62.5)')
    plt.xlabel('Wavenumber')
    plt.ylabel('Fourier Power')
    plt.title('Fourier power spectrum of temporal variations in monthly mean temp')
    plt.legend()
    plt.plot(42.4, 1.28*10**7, 'ro')
    plt.plot(42.4, 17234.8, 'ro')
    plt.savefig("PHYS640_final_P2b_fig2_Emily.png", dpi = 600)
    plt.close()

    #This was found by eyeballing.
    temporalScale_months = 512/42.4

    print('The temporal scale of this feature is', np.round(temporalScale_months, 3), 'months')

    return
''' deriv and deriv2 are each a set of 4 first order ODEs, that serve to solve
the system of two coupled second order differential equations '''
def deriv(t,y, L0, k, m,g):
    theta = y[0] #initial angle
    z1 = y[1] #initial angular velocity
    L = y[2] # #initial length
    z2 = y[3] #initial change in length with time

    thetadot = z1 #change in angle
    z1dot = (-g*np.sin(theta) - 2*z1*z2) / L
    Ldot = z2 #change in length
    z2dot = (m*L*z1**2 - k*(L-L0) + m*g*np.cos(theta)) / m
    return [thetadot, z1dot, Ldot, z2dot] #change in angle, change in angular acceleration,
    #change in length, change in acceleration of length change.

def deriv2(t,y, L0, k, m,g):
    theta = y[0] #initial angle
    z1 = y[1] #initial angular velocity
    L = y[2] # #initial length
    z2 = y[3] #initial change in length with time

    thetadot = z1 #change in anvle
    z1dot = (-g*np.sin(theta) - 2*z1*z2) / L
    Ldot = z2 #change in length
    if Ldot < L0:
        k = 100
    z2dot = (m*L*z1**2 - k*(L-L0) + m*g*np.cos(theta)) / m
    return [thetadot, z1dot, Ldot, z2dot] #change in angle, change in angular acceleration,
    #change in length, change in acceleration of length change.
    return
'''Problem
3 explores the motion of a spring pendulum '''
def problem3():
    #knowns:
    m = .5 #kg
    g = 9.81 #m/s^2
    k = 10 #N/m
    L0 = .5 #m - length of spring without weight
    L1 = L0 +m*(g/k)
    time = np.linspace(0,10,1000)
    t1 = time[0]
    tn = time[-1]

    y = [0, 0, L1, 0]
    Demo1 = scipy.integrate.solve_ivp(deriv, t_span = [t1,tn], y0 = y, args = (L0,k,m,g))
    #demonstate motionlessness
    plt.plot(Demo1.t[0:-1], Demo1.y[0,0:-1], label = "Pendulum weight")
    plt.legend()
    plt.xlabel("Time [seconds]")
    plt.ylabel("Change in angle [radians]")
    plt.title("Pendulum remains still if it starts motionless")
    plt.savefig("PHYS640_final_P3_Fig1_Emily.png", dpi = 600)
    plt.close()

    y = [0,0,L0,-np.pi/2]
    Demo2 = scipy.integrate.solve_ivp(deriv, t_span = [t1,tn], y0 = y, args = (L0,k,m,g))
    #demostrate vertical stretch
    plt.plot(Demo2.t[0:-1], Demo2.y[0,:0:-1], label = "Pendulum weight")
    plt.legend()
    plt.xlabel("Time [seconds]")
    plt.ylabel("Change in angle [radians]")
    plt.title("The Pendulum remains vertical with initial vertical stretch")
    plt.savefig("PHYS640_final_P3_Fig2_Emily.png", dpi = 600)
    plt.close()

    #demostrate vertical stretch
    plt.plot(Demo2.t[0:-1], Demo2.y[3,0:-1])
    plt.xlabel("Time [seconds]")
    plt.ylabel("Change in Spring Length with Time [m/s]")
    plt.title("Period of vertically stretched spring is ~= 1.4")
    plt.savefig("PHYS640_final_P3_Fig3_Emily.png", dpi = 600)
    plt.close()

    k = 1000
    y = [0,.1745,L0,0]
    Demo3 = scipy.integrate.solve_ivp(deriv, t_span = [t1,tn], y0 = y, args = (L0,k,m,g))
    #demonstrate large spring constant and small angle swing
    plt.plot(Demo3.t[0:-1], Demo3.y[1,0:-1])
    plt.xlabel("Time [sec]")
    plt.ylabel("Change in angle [radians]")
    plt.title("Large spring constant with small initial angle has period of ~=1.4")
    plt.savefig("PHYS640_final_P3_Fig4_Emily.png", dpi = 600)
    plt.close()

    #illistrate stiff pendulum
    ylarge = [0, 2*np.pi, L0, 0]
    partb = scipy.integrate.solve_ivp(deriv, t_span = [t1,tn], y0 = ylarge, args = (L0,k,m,g))

    plt.plot(partb.t[0:-1], partb.y[1,0:-1], label = "Large initial swing")
    plt.plot(Demo3.t[0:-1], Demo3.y[1,0:-1], label = "Small initial swing")
    plt.title("Large swing vs. Small swing of a very stiff-springed pendulum")
    plt.legend()
    plt.ylabel("Change in angle [radians]")
    plt.xlabel("Time [seconds]")
    plt.savefig("PHYS640_final_P3_Fig5_Emily.png", dpi = 600)
    plt.close()

    #use a reasonable list of initial conditions
    y = [0, np.pi/2, L0, -1]
    swing = scipy.integrate.solve_ivp(deriv, t_span = [t1,tn], y0 = y, args = (L0,k,m,g))
    theta = swing.y[0,0:-1]
    L = swing.y[2,0:-1]
    #calculate the x and z positions using trig
    x = L*np.sin(theta)
    z = -L*np.cos(theta)
    #plot the position of the pendulum weight in the x-z position
    plt.plot(x,z, label = "Pendulum weight")
    plt.legend()
    plt.xlabel("X plane position")
    plt.ylabel("Z plane position")
    plt.title("Mostion of pendulum weight in x-z plane")
    plt.savefig("PHYS640_final_P3_Fig6_Emily.png", dpi = 600)
    plt.close()

    partc = scipy.integrate.solve_ivp(deriv2, t_span = [t1,tn], y0 = y, args = (L0,k,m,g))
    partc2 = scipy.integrate.solve_ivp(deriv, t_span = [t1,tn], y0 = y, args = (L0,k,m,g))
    theta = partc.y[0,0:-1]
    L = partc.y[2,0:-1]
    #calculate the x and z positions using trig
    x = L*np.sin(theta)
    z = -L*np.cos(theta)
    theta2 = partc2.y[0,0:-1]
    L2 = partc2.y[2,0:-1]
    #calculate the x and z positions using trig
    x2 = L2*np.sin(theta2)
    z2 = -L2*np.cos(theta2)
    #plot the position of the pendulum weight in the x-z position
    plt.plot(x,z, label = "k = 100 if L below L0")
    plt.plot(x2,z2, label = "Original, k = 10")
    plt.legend()
    plt.xlabel("X position of weight")
    plt.ylabel("Z position of weight")
    plt.title("A change in K when Length of pendulum is less than L0")
    plt.savefig("PHYS640_final_P3_Fig7_Emily.png", dpi = 600)
    plt.close()


    return
''' Controls the entire program. Turn on and off functions here to run partial code'''
def main():
    N = input("Input number of layers (> 2): ")
    N = int(N)
    emissivity = input("Input the emissivity (< 1): ")
    emissivity = float(emissivity)
    print("Problem 1")
    T = problem1a()
    print("Surface temperature without a greenhouse effect is", T, "kelvin")
    T1 = problem1b(N, emissivity)
    print("The temperature of the surface with the greenhouse effect is", np.round(T1[0],3), "kelvin")
    problem1c()
    problem1d(N)
    print("Plots saved from problem 1")
    print("Problem 2")
    problem2a()
    problem2b()
    print("Plots saved from problem 2")
    print("Problem 3")
    problem3()
    print("Plots saved from problem 3")
    return

if __name__ == "__main__":
    main()
