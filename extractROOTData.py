'''
@file extractROOTData.py
@author Francesca Del Corso - francesca.del.corso@cern.ch
@date Apr 23th 2021
@version I
@brief This program extract JHARED ROOT file data using Uproot v.3 . ****** IT TAKES SOME TIME  ********

'''
import uproot3
import numpy as np
import datetime
from os import environ  

inFile = "C:\\temp\\HOUGH\\hough_pileup_truth.root"
outFile = "C:\\temp\\HOUGH\\outFileHoughPileupTruth.txt"

def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"

def extractROOTData():
        #file= uproot3.open(inFile)       
        #print(file.keys())
        #print(file.values())
        HTTHoughRootOutput=uproot3.open(inFile)["HTTHoughRootOutput"]
        #HTTHoughRootOutput.show()                              # mostra i type name e la loro interpretation

        x = HTTHoughRootOutput.array("x")                       # x is of class 'awkward0.array.jagged.JaggedArray
        y = HTTHoughRootOutput.array("y")
        #z = HTTHoughRootOutput.array( "z")
        phi = HTTHoughRootOutput.array("phi")                   # costante float
        invpt = HTTHoughRootOutput.array("invpt")               # costante float
        layer = HTTHoughRootOutput.array("layer")               # vettore di interi da 0 a 7
        #tracknumber = HTTHoughRootOutput.array("tracknumber")
        roadnumber = HTTHoughRootOutput.array("roadnumber")
        #barcode = getattr(tree ,"barcode")
        #barcodefrac = getattr(tree ,"barcodefrac")
        #eventindex = getattr(tree ,"eventindex")
        #isPixel = getattr(tree ,"isPixel")
        #isBarrel = getattr(tree ,"isBarrel")
        #etawidth = getattr(tree, "etawidth")
        #phiwidth = getattr(tree, "phiwidth")
        #etamodule = getattr(tree ,"etamodule")
        #phimodule = getattr(tree, "phimodule")
        #ID = getattr(tree, "ID")
        #candidate_barcodefrac = getattr(tree ,"candidate_barcodefrac")
        #candidate_barcode = getattr(tree ,"candidate_barcode")
        #candidate_eventindex = getattr(tree ,"candidate_eventindex")
        treeindex = HTTHoughRootOutput.array("treeindex")

        # WRITE DATA EXTRACTED INTO A FILE
        myDataFile = open(outFile, "w")
        n_points= len(x)                                        # len(x) is the maximum value 
        for i in range(n_points):                                                      
            for j in range(len(x[i])):      
                x_coord=float(x[i][j])
                y_coord=float(y[i][j]) 
                layers = int(layer[i][j])

                roadnumbers = int(roadnumber[i])
                phi_road = float(phi[i]) 
                invpt_road = float(invpt[i]) 
                treeindexes = int(treeindex[i])
              
                myDataFile.write(str(x_coord) + "," + str(y_coord) + "," + str(phi_road) + ","  + str(invpt_road)+","  + str(layers)+","  + str(roadnumbers)+","  + str(treeindexes)+"\n" )               

        myDataFile.close()
        return()

def main():  

        start = datetime.datetime.now()
        print(start)
        extractROOTData()                          
        end = datetime.datetime.now()
        print(end-start)

if (__name__ == "__main__"):
    suppress_qt_warnings()
    main()