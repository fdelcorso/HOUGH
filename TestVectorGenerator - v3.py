########################################################################################################
# Program name: Test Vector Generator 
# Author: Francesca Del Corso
# Date: 07/12/2020
# Release: II
#
# This program generates 5 line equations and for each of them it finds 8 points (x,y), for a total of 40 points. Then it generates other random points for a total of 4.000 points.
# The output shoud be a file with 4000 couples (x,y) in 16 bit binary format mixed in this way: every 8 couples must be contained in different 500 bunches. 
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
# y= mx + q  is considered a trace
# x = 32 means slope = 45°
#
########################################################################################################


import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import numpy as float16
from os import environ                                                                              # to resolve a problem in the Visual Studio Code Python environment


x_min = 32
x_max = 1023

y_min = 0
y_max = 65535

m_min = 0
m_max = 63

q_min = 0
q_max = 65535

n_lines = 5                                                                                         # number of lines I want to find 
n_points = 8                                                                                        # number of points per lines

n_tvg = 4000
    
TVGD = "C:\\temp\\TestVectorGeneratorDATA"
TVGLD = "C:\\temp\\TestVectorGeneratorLINES&DOTS"

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

def main():  
    # Creation of 3960 random (x,y) couples
    myDataFile = open(TVGD, "w+")
    for i in range(n_tvg - n_lines*n_points):
        x = '{0:016b}'.format(np.random.randint(x_min,x_max))
        y = '{0:016b}'.format(np.random.randint(y_min,y_max))
        myDataFile.write( str(x) + ", " + str(y) + "\n")
    myDataFile.close()

    

    myDataFile = open(TVGD, "r")
    contents = myDataFile.readlines()
    
    myDataFile.close()

    myLinesFile = open(TVGLD, "w")
    
    for i in range(n_lines):                                                  # trovo le 5 rette
        m = np.random.randint(m_min, m_max)                                      # m = slops vector
        q = np.random.randint(q_min, q_max)                                      # q = offsets vector
        x = np.linspace(x_min, x_max, num=x_max, dtype=int)  
        y = m*x+q                                                              
        myLinesFile.write( "y = " + str(m) + "x + " + str(q) + "\n")          # Creation of the 5 equations
        plt.plot(y, color = 'g') 

        h = 0
        while (h < 8):                                                           # per ogni retta calcolo 8 punti
            x1 = np.random.randint(x_min, x_max)
            y1 = m * x1 + q
            if (y1 <=  65535):                                                  # Contraints n. 2 
                myLinesFile.write(str('{0:016b}'.format(x1)) + ", " + str('{0:016b}'.format(y1)) + "\n")                                                                                        
                index = (495 + h) * (i + 1)                                      # Insert of the (x,y) in the right place in contents
                value = str('{0:016b}'.format(x1)) + ", " + str('{0:016b}'.format(y1)) + "\n"
                contents.insert(index, value)
                h = h + 1 
    # print(len(contents))
    myLinesFile.close()

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

     # Creation of the 4000 entries file
    
    myDataFile = open(TVGD, "w")
    contents = "".join(contents)
    myDataFile.write(contents)
    myDataFile.close()

if (__name__ == "__main__"):
    suppress_qt_warnings()
    main()