'''
@file TestVectorSimulation-v1.py
@author Francesca Del Corso - francesca.del.corso@cern.ch
@date Apr 21th 2021
@version I
@brief  Taken x,y, phi_road, invpt_road data from JAHRED ROOT files (HTTHoughOutput.root, hough_pileup_truth.root), we search for the 6,7,8 values into the accumulator (also the 67876 strings) and compare the found roads with the ones coming from Jahred files. 

CONSTRAINTS:
rho : 12 bit exadecimal in [32-4096]
phi : 16 bit exadecimal in [0-65535]
layer: 4 bit
phi_fixed = phi/2π x 2^16
rad_fixed = r/1100mm x 2^12
ROI:  Ф ∈ [0.3, 0.5], η ∈ [0.1, 0.3], q/pT ∈ [-1.0, 1.0]
https://indico.cern.ch/event/1011442/contributions/4248106/attachments/2196207/3713293/GlobalHitTestVectors.pdf 
https://gitlab.cern.ch/atlas-tdaq-ph2upgrades/atlas-tdaq-htt/tdaq-htt-offline/athena/-/blob/21.9/Trigger/TrigHTT/TrigHTTConfig/python/HTTAlgorithmTags.py#L153

'''

import numpy as np
import matplotlib.pyplot as plt
from os import environ                                                                              # to resolve a problem in the Visual Studio Code Python environment
from mpl_toolkits.mplot3d import Axes3D
import datetime
import statistics
import matplotlib as mpl


rho_min = 32                                                                                                            # ROOT file rho values in mm
rho_max = 4096             
phi_min = 0                                                                                                             # ROOT file phi values in rad
phi_max = 65535

qpt_min = -80      # -1 for Jahred  # 0 in my previous version
qpt_max = 80       # 1 for Jahred   # 63 in my previous version
phi0_min = 3000    # 0.3 in radianti
phi0_max = 5300    # 0.5 in radianti                                                                                    
xBins =  230       #6500 #230# 228         # 216 +6 +6 for Jahred    # 1200 in my previous version
yBins = 220        # 220 for Jahred            ****     NOT YET USED ****

ΔBinx = int((phi0_max-phi0_min)/xBins)  # 32 in my previous version
ΔBiny = int((qpt_max-qpt_min)/yBins)    # 1 in my previous version

n_layers = 8 

inFile = "inFileNew.txt"    
#inFile = "inFileRoad10000.txt"        
outFile8 = "C:\\temp\\HOUGH\\outFile8.txt"
outFile67876 = "C:\\temp\\HOUGH\\outFile67876.txt"
imageDir = "C:\\temp\\HOUGH\\"                                                                                          # used in accumulatorAsAnnotatedHeatmap def

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

def dataExtraction():
    rho_arr = []
    phi_arr = []
    l =[]
    phi_road_arr =[]
    invpt_road_arr = []
    phi_road_fixed_arr = []
    invpt_road_fixed_arr = []

    with open(inFile,'r') as myDataFile:
        f=myDataFile.readlines()    

    # n_points definition!!!
    n_points = len(f)

    fig = plt.figure()                
    for i in range(n_points): 
        f[i] = f[i].strip("\n").split(",") 
        x_coord = float(f[i][0])
        y_coord = float(f[i][1]) 
        layer =   int(f[i][2])                                                                                          # index=2 means LAYER in the inFile.txt if x,y,layer,roads, .. are the columns 
        phi_road =   float(f[i][4])                                                                                     # index=4 means PHI_ROAD in the inFile.txt if x,y,layer,roads, .. are the columns 
        invpt_road = float(f[i][5])                                                                                     # index=5 means INVPT_ROAD in the inFile.txt if x,y,layer,roads, .. are the columns 

        # PLOT (x,y) values 
        plt.scatter(x_coord, y_coord, color="r", alpha=0.01)

        # rho, phi CALCULATION
        rho = np.sqrt(np.square(x_coord) + np.square(y_coord))
        phi = np.arctan(y_coord/x_coord)

        # rho, phi, invpt_road, phi_road BITWISE CONVERTION              
        phi_fixed = (phi_max*phi)/6.28                                                                                  # phi_max=2^16=65535
        rho_fixed = (rho_max*rho)/1100                                                                                  # rho_max=2^12=4096
        phi_road_fixed = (phi_max*phi_road)/2*np.pi 
        invpt_road_fixed = (rho_max*invpt_road)/1100

        rho_arr.append(rho_fixed)
        phi_arr.append(phi_fixed)
        phi_road_arr.append(phi_road)
        invpt_road_arr.append(invpt_road)
        l.append(layer)    
        phi_road_fixed_arr.append(phi_road_fixed)
        invpt_road_fixed_arr.append(invpt_road_fixed)

    plt.title( str(n_points) + " (x,y) values" + "\n")
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()
    plt.close()

    myDataFile.close()  
    return (rho_arr, phi_arr, l, phi_road_fixed_arr, invpt_road_fixed_arr)                                                           

def accumulatorCreation(rho,phi,l):
    phi0=np.arange(65535)
    r_dim = len(rho)
  
    # PLOT Hough Space 
    fig = plt.figure() 
    for i in range(r_dim):
        m = (phi0 - phi[i])/rho[i]
        plt.plot(phi0,m, color="m", alpha=0.01)
    plt.title("Hough Space for " + str(r_dim) + "values" + "\n")
    plt.ylabel('qA/'r"$P_t$"'')
    plt.xlabel(''r"$φ_0$"'')
    plt.show()
    plt.close()  
    
    # ACCUMULATOR construction
    phi0 = np.arange(start=phi0_min, stop=phi0_max, step=ΔBinx)                                                         # [3000, 3010, 3020, 3030, ... , 5280,5290]
    accumulator = np.zeros( (qpt_max-qpt_min, xBins, n_layers) , dtype=np.int32)                                        # 2D array of zero, dimensions: 64x1200x8 
    K = 32 # con 64  mi trova 26 6,7,8 # 32, 128                                                                                                    # K = costant value to ajust the plot in the RoI
    for i in range(r_dim):                                                                                              # per ogni coppia (rho,phi)                    
        for j in range(xBins):                                                                                          # calcolo xBins valori di qA/Pt per rappresentare la retta nello spazio di Hough 
            qAdivPt = int( ((phi0[j] - int(phi[i]))/int(rho[i]))*K )                                                                                                                                                                                      
           
            if (qAdivPt>=qpt_min and qAdivPt< qpt_max):                                                                 # 0<=qA/Pt<=63, arrotondamento fatto all'intero piu' basso    phi = -qA/Ptr - phi0
                if (accumulator[qAdivPt+qpt_max,j,l[i]] == 0):                                                          # l'accumulatore si riempirà solo di 0 e 1
                    accumulator[qAdivPt+qpt_max,j,l[i]] +=1
                    
    # Annotated heatmap ACCUMULATOR2 construction
    accumulator2 = np.zeros( (qpt_max-qpt_min,xBins) , dtype=np.int32) 
    xBin_good = []
    qpt_good = []
    max_sum = 0
    count = 0
    for h in range(qpt_max-qpt_min):
        for k in range(xBins):
            sum=0
            for w in range(n_layers):
                if ( (accumulator[h][k][l[w]]) == 1):
                    sum +=1
            accumulator2[h][k] = sum
            if (sum > max_sum):
                    max_sum = sum
            if (sum >= 6):
                count += 1
                xBin_good.append(j)                                                                                     # row index, used for the annotated heatmap
                qpt_good.append(h)                                                                                      # column index, used for the annotated heatmap
    print("Quanti 6, 7, 8?  ",count)    
    
    # PLOT accumulator2 
    fig = plt.figure(figsize=(8,4))
    plt.imshow(accumulator2)
    plt.xlim(0, xBins)
    plt.ylim(0, qpt_max-qpt_min)

    plt.ylabel('qA/'r"$P_t$"'')                                                                     
    plt.xlabel(''r"$φ_0$"'')                                                                       
    plt.title("Accumulator (" + str(qpt_max-qpt_min) + "x" + str(xBins) + ") for " + str(r_dim) + " (r, Ф) values" + "\n")
    #plt.colorbar(img)                                                                                                  # to comment for saving a clean image
    #plt.axis('off')
    img_name = imageDir+"accumulatorRhoPhi.png"
    plt.savefig(img_name,dpi=fig.dpi, transparent=True)                                                                                                                                                       
    plt.show()
    plt.close(fig)                 
    
    #accumulatorAsAnnotatedHeatmap(accumulator2, xBin_good, qpt_good)

    return (accumulator2, max_sum)


def accumulatorAsAnnotatedHeatmap(accumulator, indici, m_trovato):
    q1 = ΔBinx*np.arange(xBins) 
    deltaIndex = 200# 60
    i_median=int(statistics.median(indici))                                                             
    m_max = max(m_trovato) + 20
    i_min = -deltaIndex+i_median
    i_max = +deltaIndex+i_median
    q2 = q1[i_min:i_max]
    mini_accumulator = accumulator[0:m_max, i_min:i_max]
    fig, ax = plt.subplots()
    im = ax.imshow(mini_accumulator)                                                                
    # Colorbar
    #cbarlabel = "Layers"
    #cbar = ax.figure.colorbar(im, ax=ax)
    #cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    #ax.set_xticks(np.arange(len(q2)))
    #ax.set_yticks(np.arange(m_max))
    #ax.set_xticklabels(q2)
    #ax.set_yticklabels(np.arange(m_max))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # text annotations
    for i in range(m_max):
        for k in range(len(q2)):
            ax.text(k, i, mini_accumulator[i, k], ha="center", va="center", color="w")
    
    #ax.set_title("Portion of the Accumulator (" + str(m_max) + "x" + str(len(q2)) +") \n")              
    fig.tight_layout()
    accumulatorAnnotated_name = "accumulatorAnnotated_"  + ".png"
    plt.savefig(imageDir+accumulatorAnnotated_name, dpi=fig.dpi, transparent=True)                                         # not good quality for the saved images.
    plt.axis('off')
    plt.show()
    plt.close()


# Research for value equal to 8 in the accumulator 
def research8(accumulator, val, phi_road, invpt_road):
    myDataFile1 = open(outFile8, "w")
    for h in range(qpt_max-qpt_min):
        for k in range(xBins):                                                                  
                if ( (accumulator[h,k] >= 6)):
                    print("Found "+str(accumulator[h,k]))
                    print(" qA/Pt = "+ str(h+qpt_min) + ", phi0 = " + str(phi0_min+ΔBinx*(k)))
                    
                    myDataFile1.write("Found "+str(accumulator[h,k]) )
                    myDataFile1.write(" qA/Pt = "+ str(h - abs(qpt_min)) + ", phi0 = " + str(phi0_min+ΔBinx*(k)) + '\n')     # The offset abs(qpt_min) is subtracted

    # COMPARAZIONE CON LE ROAD DI JARED
    # to do ....

    myDataFile1.close()

# Research for the string (>6)(>7)(=8)(>7)(>6)
def researchString67876(accumulator):
    myDataFile1 = open(outFile67876, "w")
    for h in range(qpt_max-qpt_min):
        for k in range(xBins):
            if ((k+4) < xBins):                                                                     
                if ( ( accumulator[h,k] >= 6) and (accumulator[h,k+1] >= 7) and (accumulator[h,k+2] == 8)  and (accumulator[h,k+3] >= 7) and (accumulator[h,k+4] >= 6) ):
                    print("Found "+str(accumulator[h,k]) +str(accumulator[h,k+1])+str(accumulator[h,k+2])+str(accumulator[h,k+3])+str(accumulator[h,k+4]))
                    print(" qA/Pt = "+ str(h- abs(qpt_min)) + ", phi0 = " + str(phi0_min+ΔBinx*(k+2)))
                    myDataFile1.write("Found string "+str(accumulator[h,k]) +str(accumulator[h,k+1])+str(accumulator[h,k+2])+str(accumulator[h,k+3])+str(accumulator[h,k+4]))
                    myDataFile1.write(" qA/Pt = " + str(h- abs(qpt_min)) + ", phi0 = " + str(phi0_min+ΔBinx*(k+2)) + '\n')            # The offset abs(qpt_min) is subtracted    
    myDataFile1.close()

def main():  

        start = datetime.datetime.now()
        print(start)

        rho, phi, l, phi_road, invpt_road = dataExtraction()    
        accumulator, val= accumulatorCreation(rho,phi,l)  
        research8(accumulator, val, phi_road, invpt_road)
        researchString67876(accumulator)

        end = datetime.datetime.now()
        print(end-start)

if (__name__ == "__main__"):
    suppress_qt_warnings()
    main()