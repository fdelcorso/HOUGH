'''
Program name: Test Vector Generator for HTTHoughOutput
Author: Francesca Del Corso
Date: 25/03/2021
Release: XIII

Given a data extraction from JHARED  Hough_pileup_truth_file.root file (outFile2.txt)), this program plots rho and phi, We don't consider z component.
ROI:  Ф ∈ [0.3, 0.5], η ∈ [0.1, 0.3], q/pT ∈ [-1.0, 1.0]

'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from numpy import random
from os import environ                                   # to resolve a problem in the Visual Studio Code Python environment
import statistics
from PIL import Image
from random import randint, seed
import pandas as pd
import matplotlib.cm as cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D


file = "outFile2.txt"                                   # in C:\Users\delcorso\OneDrive - Istituto Nazionale di Fisica Nucleare\INFN-BO\LAVORI\Dottorato di Ricerca in Fisica\HOUGH
file1 = "TestVectorGeneratorRESULTS"
n_points= 30000# 5000    # max value: len(f)

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"
    
def dataExtraction():
    x0 = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    x5 = []
    x6 = []
    x7 = []

    y0 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    y5 = []
    y6 = []
    y7 = []
   
    rho0 = []
    rho1 = []
    rho2 = []
    rho3 = []
    rho4 = []
    rho5 = []
    rho6 = []
    rho7 = []

    phi0 = []
    phi1 = []
    phi2 = []
    phi3 = []
    phi4 = []
    phi5 = []
    phi6 = []
    phi7 = []

    #z = []
    #l =[]
    #colors = ['r', 'g', 'b', 'r', 'm', 'k', 'c', 'y']
    #cmap = cm.get_cmap(name='rainbow')                  # da provare
    #mpl.rcParams['legend.fontsize'] = 10

    fig = plt.figure()
    ax = fig.add_subplot(111)
    #ax = fig.gca(projection='3d')
    ax.set_xlabel('rho [mm]')
    ax.set_ylabel('phi [rad]')
    #ax.set_zlabel('z')
    ax.set_xlim(200, 1200)
    #ax.set_ylim(0, 0.0001)
    #ax.set_zlim(-300, 300)

    with open(file,'r') as myDataFile:
        f=myDataFile.readlines()                                                                    
   
    n_points = len(f)                                  
    for i in range(n_points): 
        f[i] = f[i].strip("\n").split(",")           # strip: rimuove il carattere \n, split divide una stringa in una lista di stringhe prendendo come separatore la ,
        if (int(f[i][3]) == 0):
            x_0=float(f[i][0])
            y_0=float(f[i][1])
        #z_coord=int(float(f[i][2]))                 # ignoriamo per il momento la coordinata z
            x0.append(x_0)                                                                  
            y0.append(y_0)   
        #z.append( int(float(f[i][2])) )     
        #l.append( int(f[i][3]))       
            rho0.append(np.sqrt(np.square(x_0) + np.square(y_0))) 
            phi0.append(np.arctan(y_0/x_0))
        else:
            if (int(f[i][3]) == 7):
                x_7=float(f[i][0])
                y_7=float(f[i][1])
                x7.append(x_7)                                                                  
                y7.append(y_7)   
                rho7.append(np.sqrt(np.square(x_7) + np.square(y_7))) 
                phi7.append(np.arctan(y_7/x_7))
            else:
                if (int(f[i][3]) == 6):
                    x_6=float(f[i][0])
                    y_6=float(f[i][1])
                    x6.append(x_6)                                                                  
                    y6.append(y_6)   
                    rho6.append(np.sqrt(np.square(x_6) + np.square(y_6))) 
                    phi6.append(np.arctan(y_6/x_6)) 
                else:
                    if (int(f[i][3]) == 5):
                        x_5=float(f[i][0])
                        y_5=float(f[i][1])
                        x6.append(x_5)                                                                  
                        y5.append(y_5)   
                        rho5.append(np.sqrt(np.square(x_5) + np.square(y_5)))
                        phi5.append(np.arctan(y_5/x_5))  
                    else:
                        if (int(f[i][3]) == 4):
                            x_4=float(f[i][0])
                            y_4=float(f[i][1])
                            x4.append(x_4)                                                                  
                            y4.append(y_4)   
                            rho4.append(np.sqrt(np.square(x_4) + np.square(y_4))) 
                            phi4.append(np.arctan(y_4/x_4)) 
                        else:
                            if (int(f[i][3]) == 3):
                                x_3=float(f[i][0])
                                y_3=float(f[i][1])
                                x3.append(x_3)                                                                  
                                y3.append(y_3)   
                                rho3.append(np.sqrt(np.square(x_3) + np.square(y_3))) 
                                phi3.append(np.arctan(y_3/x_3))    
                            else:
                                if (int(f[i][3]) == 2):
                                    x_2=float(f[i][0])
                                    y_2=float(f[i][1])
                                    x2.append(x_2)                                                                  
                                    y2.append(y_2)   
                                    rho2.append(np.sqrt(np.square(x_2) + np.square(y_2))) 
                                    phi2.append(np.arctan(y_2/x_2)) 
                                else:
                                    if (int(f[i][3]) == 1):
                                        x_1=float(f[i][0])
                                        y_1=float(f[i][1])
                                        x1.append(x_1)                                                                  
                                        y1.append(y_1)   
                                        rho1.append(np.sqrt(np.square(x_1) + np.square(y_1))) 
                                        phi1.append(np.arctan(y_1/x_1))                                               

    #print(np.arctan(y_0/x_0))
    #plt.scatter(x, y, z, alpha=0.01, c='r') 
    plt.scatter(rho0, phi0, c='r') 
    plt.scatter(rho1, phi1, c='y') 
    plt.scatter(rho2, phi2, c='k') 
    plt.scatter(rho3, phi3, c='pink') 
    plt.scatter(rho4, phi4, c='gray') 
    plt.scatter(rho5, phi5, c='m' ) 
    plt.scatter(rho6, phi6, c='c' ) 
    plt.scatter(rho7, phi7, c='b' ) 
    plt.title(str(n_points) + " (x,y) values from hough_pileup_truth.root" + "\n")
    plt.show()
    plt.close()
    
    myDataFile.close()  
    #return (rho, phi, l)


def main():  

        dataExtraction()    
       

if (__name__ == "__main__"):
    suppress_qt_warnings()
    main()