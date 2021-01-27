########################################################################################################
# Program name: Test Vector Generator
# Author: Francesca Del Corso
# Date: 25/01/2021
# Release: VII
# 
# 1 retta e 16 valori (x,y) t.c. x è stato preso negli intorni +-deltaX di multipli di deltaBinX.
# Dati m, q random trovo 1 retta che passi per le 16 coppie (x,y)  con y>0. Space parameter representation and accumulator.
#
# Constraints:  
# 1. x must be an INT in the range [32 - 1023]          It is rho (dist from (0,0) in polar coordinates)            (16 bits)  
# 2. y must be an INT in the range [0 - (2exp16 -1)]    It is phi                                                   (16 bits)
# 3. m (slope) must be an INT in the range [0-63]  It is qA/Pt                                                      (6 bits) 
# 4. q (offset) must be an INT in the range [0 - (2exp16 -1)]                                                       (16 bits)
# 5. in the final file numbers must be mixed.
# 6. qm -ql > 31 for every m, l (the lines must have offsets > 31 otherwise it is difficult to distinguish among them) (NOT YET IMPLEMENTED)
# 7. mk - mh > 0 for every k, h (slitely difference in slopes are not permitted if constraints n. 6 is false)     (NOT YET IMPLEMENTED)      
# 8. every (x,y) of the 5 lines must be contained in the every 8 blocks of 500 entries 
#
# Notes: 
# qA/Pt : A is a costant, q is the charge, Pt= trasverse moment 
# slope: 0-1023 (2exp10 -1), phi = 2ext16 -1 (0-360°) phi=38°
# y= -mx + q  is the equation
# x = 32 means slope = 45°
#
########################################################################################################

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from os import environ                                                                              # to resolve a problem in the Visual Studio Code Python environment

x_min = 32
x_max = 1023             
y_min = 0
y_max = 65535
m_min = 0
m_max = 63
q_min = 10000
q_max = 20000                                                                                       # 65535 come rappresentazione, in realtà sarebbe nBin*deltaBin=1200x32=38400
n_lines = 1                                                                                         # number of lines I want to find 
n_points = 16  
n_layers = 8                                                                                     # number of points per lines
deltaBinX = 128                                                                                     # to define the interval into which to take the x points
deltaX = 5                                                                                          # +-deltaX for the interval
nBin = 1200
deltaBin=32


TVGD = "C:\\temp\\TestVectorGeneratorDATA"
TVGLD = "C:\\temp\\TestVectorGeneratorLINES&DOTS"
TVGD1 = "C:\\temp\\TestVectorGeneratorDATA1"

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

# Calcolo m_calcolato = (q_trovato -y[i])/x[i] e lo confronto con m_trovato per vedere se i valori trovati soddisfano l'equazione della retta m=(q-y)/x
def checkCalcolato(x, y, q_trovato, m_trovato):
    for h in range(len(x)):  
        num = 0
        for k in range(len(q_trovato)):
            m_calcolato = int( (q_trovato[k] - y[h])/x[h] )
            if (m_calcolato == m_trovato[k]):
                #print("Found! y = -"+str(m_calcolato) + "x + " + str(q_trovato[k]))
                num +=1                 # usato se commento il print

def accumulatorCreation(x,y):
    # Accumulator creation
    
    q1 = deltaBin*np.arange(nBin)                                                   # q1 = [0,32,64,96,128,160,192,224,256,288, ..., deltaBin*nBin-1]                                         
    q_trovato = []   
    m_trovato = []  
    k=0      
    count = 0 
    accumulator = np.zeros( (2*deltaBin, nBin) , dtype=np.int32)                    # list of lists
             
    while (count < 8) :        
        trovato  = np.zeros( (2*deltaBin, nBin) , dtype=np.int32)                   # matrice di appoggio 64x1200                                                                              
        for i in range(k, k+2):                                                         # le coppie appartengono a due a due ai diversi layers                             
            for j in range(nBin):                                                       # per ogni coppia (x,y) calcolo nBin valori di m 
                m = int(q1[j]/x[i] - y[i]/x[i])                                     # 0<=m<=63 arrotondamento fatto all'intero piu' basso   
                if (m>m_min and m<= m_max):    
                    if (trovato[m,j] == 0):
                        accumulator[m,j] += 1
                        trovato[m,j] += 1
                    if (accumulator[m,j] > 7): 
                        m_trovato.append(m)
                        q_trovato.append(q1[j])
                        # print("y = -" + str(m) + "x + " + str(q1[j]) 
        k += 2
        count +=1

    # Accumulator representation 
    fig = plt.figure(figsize=(8,4))                                                 # taglia del plot
    plt.imshow(accumulator)   
    plt.xlim(0, 1200)
    plt.ylim(0, m_max)
    plt.ylabel('m')
    plt.xlabel('j')
    plt.title("Accumulator (" + str(m_max) + "x" + str(nBin) + ") \n")
    plt.colorbar()
    plt.show()

    #accumulatorAsHeatmap(accumulator, q1)
    return (q_trovato, m_trovato)

def accumulatorAsHeatmap(accumulator, q1):
# Accumulator representation as heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(accumulator)
    
    # Colorbar
    cbarlabel = "Layers"
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    ax.set_xticks(np.arange(nBin))
    ax.set_yticks(np.arange(m_max))

    ax.set_xticklabels(q1)
    ax.set_yticklabels(np.arange(m_max))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # text annotations
    for i in range(m_max):
        for k in range(nBin):
            text = ax.text(k, i, accumulator[i, k], ha="center", va="center", color="w")
    
    ax.set_title("Accumulator (" + str(m_max) + "x" + str(nBin) +") \n")               # 64x1200 al max
    fig.tight_layout()
    
    plt.show()


def dataGenerator():
    myDataFile = open(TVGD, "w")
    myLinesFile = open(TVGLD, "w")
    myDataFile1 = open(TVGD1, "w")

    # x vector  
    x = []
    x1 = np.arange(start=deltaBinX, stop=x_max+deltaBinX, step=deltaBinX)                            # [128,256,384,512,640,768,896,1024]

    # I versione
    count=0
    trovato =[]
    while (count<16):
        for i in range(len(x1)):
            trovato = np.random.randint(-deltaX+x1[i], deltaX+x1[i], size=2)
            for j in range(len(trovato)):
                if (trovato[j] < 1024) and (count<16) and (trovato[j] not in x ):
                    x.append(trovato[j])
                    count +=1

    # II versione
    #for i in range(len(x1)):
    #    count = 0
    #    while(count<2):
    #            trovato = np.random.randint(-deltaX+x1[i], deltaX+x1[i])
    #            if ((trovato < 1024) and (trovato not in x)):
    #                x.append(trovato)
    #                count +=1
                   
    print("x=" + str(x))

    # m,q vectors
    k=1
    y =[]                                                                                               # y is possible unbound if I miss this line
    while (k):
        m = np.random.randint(m_min, m_max, size=1)                                                     # m = slops vector
        q = np.random.randint(q_min, q_max, size=1)                                                     # q = offsets vector
        y = -m* x + q
        if (y > 0).all():
            k=0
            retta = "y=-"+str(m[0])+"x+" + str(q[0])
            y = np.array(y).tolist()                                                                    
            print("y=" + str(y))
            print("Generated line in (x,y) space: " + str(retta))
            # Plot line
            myLinesFile.write( "y = -" + str(m[0]) + "x + " + str(q[0]) + "\n")                     
            plt.plot(x,y, label=retta)    

    for j in range(len(x)):
        plt.scatter(x[j], y[j])    
        myLinesFile.write(str('{0:016b}'.format(x[j])) + ", " + str('{0:016b}'.format(y[j])) + "\n")
        myDataFile1.write(str(x[j]) + ", " + str(y[j]) + "\n")   
        myDataFile.write(str('{0:016b}'.format(x[j])) + ", " + str('{0:016b}'.format(y[j])) + "\n")      

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title('Generated lines in (x,y)\n' )
    plt.legend()
    plt.savefig("C:\\temp\\image.png")                                                              # per memorizzare il plot su file
    plt.show()

    myLinesFile.close()
    myDataFile.close()

def HoughSpaceGenerator():
    x = []
    y = []
    with open(TVGD,'r') as myDataFile:
        f=myDataFile.readlines()                                                                    # f is a list

    for i in range(len(f)):     
        f[i] = f[i].replace(',','').strip("\n").split()
        x.append( int(f[i][0], 2))                                                                  
        y.append( int(f[i][1], 2))                                                                     

    # Plot lines in Parameter Space                                                                #se x<=deltaBin devo calcolare q e usare la formula q=mx+y; se x>=deltaBin utilizzo la formula m=(q-y)/x
    q = np.arange(q_max)
    for i in range(len(x)):
        m = q/x[i] - y[i]/x[i]
        retta = 'm=-q/'+str(x[i])+" + " +str(y[i])+"/"+str(x[i])
        plt.plot(q,m)

    plt.title("Parameter Space")
    plt.ylabel('m')
    plt.xlabel('q')
    plt.show()

    q_trovato, m_trovato = accumulatorCreation(x,y)
    checkCalcolato(x, y, q_trovato, m_trovato)

    myDataFile.close()  

def main():  
    dataGenerator()
    HoughSpaceGenerator()
    

if (__name__ == "__main__"):
    suppress_qt_warnings()
    main()