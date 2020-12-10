########################################################################################################
# Program name: Test Vector Generator 
# Author: Francesca Del Corso
# Date: 25/11/2020
# Release: I
#
# This program generates 5 line equations and for each of them it finds 8 points (x,y), for a total of 40 points. Then it generates other random points for a total of 4.000 points.
# The output shoud be a file with 4000 mixed couples (x,y) in 16 bit binary format. 
#
#
# Constraints:  
# 1. x must be an INT in the range [32 - 1023]          It is rho (dist from (0,0) in polar coordinates)            (16 bits)
# 2. y must be an INT in the range [0 - (2exp16 -1)]    It is phi                                                   (16 bits)
# 3. m (slope) must be an INT in the range [0-63]  It is qA/Pt                                                      (6 bits) 
# 4. q (offset) must be an INT in the range [0 - (2exp16 -1)]                                                       (16 bits)
# 5. in the final file numbers must be mixed.
# 6. qm -ql > 31 for every m, l (the lines must have offsets > 31 otherwise it is difficult to distinguish among them) (NOT YET IMPLEMENTED)
# 7. mk - mh > 0 for every k, h (slitely difference in slopes are not permitted if constraints n. 6 is false)     (NOT YET IMPLEMENTED)      
#
# Notation: 
# qA/Pt : A is a costant, q is the charge, Pt= trasverse moment 
# slope: 0-1023 (2exp10 -1), phi = 2ext16 -1 (0-360°) phi=38°
# y= mx + q  is considerd a trace
#
########################################################################################################


import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import numpy as float16



x_min = 32
x_max = 1023

y_min = 0
y_max = 65535

m_min = 0
m_max = 63

q_min = 0
q_max = 65535

n_lines = 5                                                             # number of lines I want to drow 
n_points = 8                                                            # number of points per lines

n_tvg = 4000
    

TVGD = "C:\\temp\\TestVectorGeneratorDATA"
TVGLD = "C:\\temp\\TestVectorGeneratorLINES&DOTS"


def randomize (f):
    lines = open(TVGD).readlines()
    random.shuffle(lines)
    open(TVGD, 'w').writelines(lines)


def main():
    h = 0
    f = open(TVGD, "w")
    f1 = open(TVGLD, "w")
    m = np.random.randint(m_min, m_max, size=n_lines*n_points)          # m = slops vector
    q = np.random.randint(q_min, q_max, size=n_lines*n_points)          # q = offsets vector
    x = np.linspace(x_min, x_max, num=n_lines*n_points, dtype=int) 

    for i in range(len(x)):
        y = m[i]*x+q[i]  
        if ((y[i] < 65535) and (h < 5)):                                 # y must be < 65535 and INT  (Contraints n. 2) 
            plt.plot(y, color='r')                                       # Not necessary                     
            f1.write( "y = " + str(m[i]) + "x + " + str(q[i]) + "\n")    # Creation of TGVLD file with the 5 equations of lines
            for x1 in range(8):                                          # Creation of 8 couples (x,y) for every line 
                y1 = m[i]*x1+q[i]
                f.write(str('{0:016b}'.format(x1)) + ", " + str('{0:016b}'.format(y1)) + "\n")
                f1.write(str('{0:016b}'.format(x1)) + ", " + str('{0:016b}'.format(y1)) + "\n")
        h += 1
                                        
    plt.ylabel('y')
    plt.xlabel('x')
    plt.show()

    # Creation of (n_tvg - n_lines) random (x,y) couples
    for i in range(n_tvg - n_lines*n_points):
        x = '{0:016b}'.format(np.random.randint(x_min,x_max))
        y = '{0:016b}'.format(np.random.randint(y_min,y_max))
        f.write( str(x) + ", " + str(y) + "\n")

    randomize(f)                                                        # Constraint n. 5

    f.close()
    f1.close()

if (__name__ == "__main__"):
     main()