'''
@file TV-GenSim.py
@author Francesca Del Corso - francesca.del.corso@cern.ch
@date 13 dic 2021
@
@brief  Taken rho and phi from my inFile_0.txt file,  we search for ROADS using accumulator values >= treshold
Statistics: al variare del numero di roads iniziali e di fake cluster, vengono calcolati i valori di extracted, exceeded, found (good=equal to the original ones) roads, % di successo 
e di densità di riempimento percentuale dell'accumulatore. I grafici vengono fatti disegnare dal programma TV-Stats.py

To Do List
1. Andrebbe utilizzato un dataframe per gestire i valori delle colonne del file di input (rho, phi, layer, roadnumber, phi0, invpt, tree_index, fakeORnot)
2. mescolare i dati nel file di input


'''



import numpy as np
import matplotlib.pyplot as plt
from os import environ                                                             # to resolve a problem in the Visual Studio Code Python environment
import datetime
import statistics
import random
import json
from termcolor import cprint
from matplotlib.axis import Axis
import shutil


image_dir = "image\\"
outFile   = "TVFoundRoads.txt"      # it contains info about found roads
outFile1  = "TVGen_r.txt"           # it contains only data belonging to roads 
outFile2  = "TVGen.txt"             # it contains data belonging to roads plus fake clusters


with open("TV-conf.json") as json_data_file:     
    data = json.load(json_data_file)

# qinvpt
qinvpt_min          = float(data["TV-Gen_Tags"]["qinvpt_min"])                            #-0.12 #-0.03  #-1    
qinvpt_bins         = int(data["TV-Gen_Tags"]["qinvpt_bins"])                             # ex yBins 
grana_qinvpt        = float(data["TV-Gen_Tags"]["grana_qinvpt"] )                         # 0.0015 fornisce la granularità delle rette
qinvpt_max          = qinvpt_min + qinvpt_bins*grana_qinvpt
deltaBinQinvpt      = (qinvpt_max-qinvpt_min)/qinvpt_bins    

# phi, phi0
phi0_min            = float(data["TV-Gen_Tags"]["phi0_min"] ) 
phi0_bins           = int(data["TV-Gen_Tags"]["phi0_bins"])                              # ex xBins 
grana_phi0          = float(data["TV-Gen_Tags"]["grana_phi0"] )                          # 0.0015 fornisce la granularità delle rette
phi0_max            = phi0_min + phi0_bins*grana_phi0
deltaBinPhi0        = (phi0_max-phi0_min)/phi0_bins                                                                                                                        

# rho
delta_r             = float(data["TV-Gen_Tags"]["delta_rho"] )                       
r0_centr            = int(data["TV-Gen_Tags"]["r0_centr"] )

r0_min              = r0_centr - r0_centr*delta_r   # circa 120
r0_max              = r0_centr + r0_centr*delta_r   # circa 160

r1_centr            = int(data["TV-Gen_Tags"]["r1_centr"] )
r1_min              = r1_centr - r1_centr*delta_r   # circa 240
r1_max              = r1_centr + r1_centr*delta_r   # circa 280

r2_centr            = int(data["TV-Gen_Tags"]["r2_centr"] )
r2_min              = r2_centr - r2_centr*delta_r   # circa 360
r2_max              = r2_centr + r2_centr*delta_r   # circa 400

r3_centr            = int(data["TV-Gen_Tags"]["r3_centr"] )
r3_min              = r3_centr - r3_centr*delta_r   # circa 480
r3_max              = r3_centr + r3_centr*delta_r   # circa 520

r4_centr            = int(data["TV-Gen_Tags"]["r4_centr"] )
r4_min              = r4_centr - r4_centr*delta_r   # circa 600
r4_max              = r4_centr + r4_centr*delta_r   # circa 640

r5_centr            = int(data["TV-Gen_Tags"]["r5_centr"] )
r5_min              = r5_centr - r5_centr*delta_r   # circa 720
r5_max              = r5_centr + r5_centr*delta_r   # circa 760

r6_centr            = int(data["TV-Gen_Tags"]["r6_centr"] )
r6_min              = r6_centr - r6_centr*delta_r   # circa 840
r6_max              = r6_centr + r6_centr*delta_r   # circa 880

r7_centr            = int(data["TV-Gen_Tags"]["r7_centr"] )
r7_min              = r7_centr - r7_centr*delta_r   # circa 960
r7_max              = r7_centr + r7_centr*delta_r   # circa 1000

# Used to generate data
phi0_centr          = float(data["TV-Gen_Tags"]["phi0_centr"])
delta_phi0          = float(data["TV-Gen_Tags"]["delta_phi0"])
phi0_max_road       = phi0_centr*delta_phi0 


A                   = float(data["TV-Gen_Tags"]["A"] )                     # A=3.10^(-4) =0.0003         
n_layers            = int(data["TV-Gen_Tags"]["n_layers"] )   
event_num           = int(data["TV-Gen_Tags"]["event_num"] )    
file_generation_number = int(data["TV-Gen_Tags"]["file_generation_number"] ) 
n_cluster_per_layer = int(data["TV-Gen_Tags"]["n_cluster_per_layer"] )  
threshold           = int(data["TV-Sim_Tags"]["threshold"] )                        
hitExtend_x         = data["TV-Sim_Tags"]["hitExtend_x"]  

# Fake data                                                                                                                                                                                                                                                
treeindex_fake_cluster = int(data["TV-Gen_Tags"]["treeindex_fake_cluster"] )                                                                                                                            
#fake_clusters = int(data["TV-Gen_Tags"]["fake_clusters"] )

# seed
seed = int(data["TV-Gen_Tags"]["seed"] )
random.seed(seed)  


def suppress_qt_warnings():
    environ["QT_DEVICE_PIXEL_RATIO"] = "0"
    environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    environ["QT_SCREEN_SCALE_FACTORS"] = "1"
    environ["QT_SCALE_FACTOR"] = "1"


def TV_Gen_r(roads):
    f = open(outFile1, "w")
    for treeindex in range(event_num):

        for road_number in range(roads):
            # Given the (phi0,invpt) values belonging to a road, we calculate the (r,phi) for each of the (event_number*#roads) clusters
            invpt = random.uniform(qinvpt_min, qinvpt_max)  
            unif_min = phi0_min + ((phi0_max-phi0_min)/(roads+1))*(road_number+1) - phi0_max_road
            unif_max = phi0_min + ((phi0_max-phi0_min)/(roads+1))*(road_number+1) + phi0_max_road
            phi0  = random.uniform(unif_min, unif_max)         

            cluster_per_layer = 0
            while cluster_per_layer < n_cluster_per_layer:
                
                # calcolo rho
                r0 = random.uniform(r0_min, r0_max) 
                r1 = random.uniform(r1_min, r1_max) 
                r2 = random.uniform(r2_min, r2_max) 
                r3 = random.uniform(r3_min, r3_max) 
                r4 = random.uniform(r4_min, r4_max) 
                r5 = random.uniform(r5_min, r5_max) 
                r6 = random.uniform(r6_min, r6_max) 
                r7 = random.uniform(r7_min, r7_max) 

                r = [r0,r1,r2,r3,r4,r5,r6,r7]

                # Calcolo phi
                for i in range(n_layers):
                                        
                    phi = phi0 - A*invpt*r[i]
                    
                    # Scrittura su file dei risultati
                    f.write(str(r[i]) + "," + str(phi) +  "," + str(i) +  "," + str(road_number) + "," + str(phi0) +  "," + str(invpt) +  "," + str(treeindex) + "," + str("Y") + "\n" )
                
                cluster_per_layer += 1
    f.close()

# Fake clusters creation 
def TV_Gen_fc(fake_clusters):
    f = open(outFile2, "a")
    for j in range (fake_clusters):
    
        rho = random.uniform(*random.choice([(r0_min, r0_max), (r1_min, r1_max), (r2_min, r2_max), (r3_min, r3_max), (r4_min, r4_max), (r5_min, r5_max), (r6_min, r6_max), (r7_min, r7_max)]))
        
        if (rho >= r0_min) and (rho <= r0_max):
            layer = 0
        elif (rho >= r1_min) and (rho <= r1_max):
            layer = 1
        elif (rho >= r2_min) and (rho <= r2_max):
            layer = 2
        elif (rho >= r3_min) and (rho <= r3_max):
            layer = 3
        elif (rho >= r4_min) and (rho <= r4_max):
            layer = 4
        elif (rho >= r5_min) and (rho <= r5_max):
            layer = 5
        elif (rho >= r6_min) and (rho <= r6_max):
            layer = 6
        elif (rho >= r7_min) and (rho <= r7_max):
            layer = 7

        phi0_fc = random.uniform(phi0_min,phi0_max)
        invpt_fc = random.uniform(qinvpt_min, qinvpt_max) 
        phi_fake_cluster = phi0_fc - A*invpt_fc*rho

        road_num_fake_cluster   = random.randint(1000,2000)                 # numero che viene assegnato alla road
        phi0_fake_cluster       = random.uniform(phi0_min,phi0_max)
        invt_fake_cluster       = random.uniform(qinvpt_min,qinvpt_max)

        #f (j < fake_clusters-1):
        f.write(str(rho) + "," + str(phi_fake_cluster) +  "," + str(layer) +  "," + str(road_num_fake_cluster) + "," + str(phi0_fake_cluster) +  "," + str(invt_fake_cluster) +  "," + str(treeindex_fake_cluster) + "," + str("N") + "\n" )
        #else:
        #   f.write(str(rho) + "," + str(phi_fake_cluster) +  "," + str(layer) +  "," + str(road_num_fake_cluster) + "," + str(phi0_fake_cluster) +  "," + str(invt_fake_cluster) +  "," + str(treeindex_fake_cluster) + "," + str("N") )

    
    f.close()
    return(outFile2)


# USARE I DATAFRAME PER OTTENERE I VETTORI DEI VALORI DA CSV FILE!
def dataExtraction(outFile2):
    rho_arr = []
    phi_arr = []
    l =[]
    phi_road_arr =[]
    invpt_road_arr = []
    tree_index_arr = []
    road_number_arr = []
    fakeORnot_arr = []

    with open(outFile2,'r') as myDataFile:
        f=myDataFile.readlines()    

    # n_points definition
    n_points = len(f)

    fig = plt.figure()                
    for i in range(n_points):                               
        f[i] = f[i].strip("\n").split(",") 
        rho = float(f[i][0])
        phi = float(f[i][1]) 
        layer =   int(f[i][2])                                             
        road_number = int(f[i][3])
        phi_road =   float(f[i][4])                                                                                     
        invpt_road = float(f[i][5])                                                                                    
        tree_index = int(f[i][6]) 
        fakeORnot = str(f[i][7])                                                                                                                                                                  
        
        # PLOT (x,y) values 
        plt.scatter(rho, phi, color="r", alpha=0.2)
       
        phi_arr.append(phi)
        rho_arr.append(rho)
        l.append(layer) 
        phi_road_arr.append(phi_road)
        invpt_road_arr.append(invpt_road)
        road_number_arr.append(road_number)
        tree_index_arr.append(tree_index)
        fakeORnot_arr.append(fakeORnot)

    plt.title( str(n_points) + " (r,Φ) values" + "\n")
    plt.ylabel('r')
    plt.xlabel('Φ')
    #plt.show()
    plt.close()

    myDataFile.close()  

    return (phi_arr, rho_arr, l, phi_road_arr, invpt_road_arr, fakeORnot_arr, road_number_arr)                                                           

def accumulatorCreation(phi,rho,l, road_num, fake_num):
   
    # Hough Space PLOT 
    fig = plt.figure() 
    phi0_arr=np.arange(phi0_min, phi0_max, deltaBinPhi0)                                               
    for i in range(len(rho)):
        qAOverPt = (phi0_arr - phi[i])/rho[i]
        plt.plot(phi0_arr,qAOverPt, color="k", alpha=0.1)

    plt.title("Hough Space for " + str(len(rho)) + " (r, Φ) values" + "\n")
    plt.ylabel('Aq/'r"$P_t$ (GeV-1)"'')
    plt.xlabel(''r"$Φ_0$ (rad)"'')
    #plt.show()
    plt.close()  
    
    # ACCUMULATOR construction
    invpt_arr = np.arange(qinvpt_min,qinvpt_max,deltaBinQinvpt)     
    phi0_arr=np.arange(phi0_min,phi0_max,deltaBinPhi0)         # len(phi0)=xBins                                                                      
    accumulator = np.zeros((qinvpt_bins, phi0_bins, n_layers), dtype=np.int32)                            # 3D array   216X216X8                                                                                                           
    
    for i in range(len(rho)):   
        # se  r>deltaBinPhi0/deltaBinQinvpt => uso la formula phi0=phi+r*qAPT, se r<deltaBinPhi0/deltaBinQinvpt => uso la formula qAPt=(phi0-phi)/r      (questo per evitare i buchi sulle rette nell'accumulatore)                                                             
        if (rho[i] <= deltaBinPhi0/deltaBinQinvpt):
            for j in range(qinvpt_bins):                    
                phi0 = 8*invpt_arr[j]*rho[i] + phi[i]                                                            
                if (phi0>=phi0_min and phi0<phi0_max):  
                        h = int((phi0-phi0_min)/deltaBinPhi0)                                                                    
                        if (accumulator[j,h,l[i]] == 0): 
                            accumulator[j,h,l[i]] +=1 
                            # HitExtend
                            hitExtend_bins=int(hitExtend_x[l[i]])
                            if  (hitExtend_bins != 0):
                                for k in range(1, hitExtend_bins+1):
                                    if ( ((h-k)>= 0) and ((h+k)<phi0_bins) ):
                                        if (accumulator[j,h-k,l[i]] == 0):
                                            accumulator[j,h-k,l[i]] +=1    
                                        if (accumulator[j,h+k,l[i]] == 0):
                                            accumulator[j,h+k,l[i]] +=1
    
        else:
            for w in range(phi0_bins):  
                invpt = (phi0_arr[w] - phi[i])/(A*rho[i])                                                     
                if (invpt>=qinvpt_min and invpt<qinvpt_max):  
                        h = int((invpt-qinvpt_min)/deltaBinQinvpt)                                                                    
                        if (accumulator[h,w,l[i]] == 0): 
                            accumulator[h,w,l[i]] +=1 
                            
                            # HitExtend
                            hitExtend_bins=int(hitExtend_x[l[i]])
                        
                            if  (hitExtend_bins != 0):
                                for k in range(1, hitExtend_bins+1):
                                    if ( ((h-k)>= 0) and ((h+k)<qinvpt_bins) ):
                                        if (accumulator[h-k,w,l[i]] == 0):
                                            accumulator[h-k,w,l[i]] +=1    
                                        if (accumulator[h+k,w,l[i]] == 0):
                                            accumulator[h+k,w,l[i]] +=1
   
    # Annotated heatmap ACCUMULATOR2 construction
    accumulator2 = np.zeros((qinvpt_bins,phi0_bins), dtype=np.int32)                                      # 216X216
    phi0_bins_good = []
    qpt_bin_good = []
    max_sum = 0
    count = 0
    for h in range(qinvpt_bins):
        for k in range(phi0_bins):
            sum=0
            for w in range(n_layers):
                if ( accumulator[h,k,w] == 1):
                    sum +=1
            accumulator2[h,k] = sum
            if (sum > max_sum):
                    max_sum = sum
            if (sum >= threshold):
                count += 1
                phi0_bins_good.append(k)       # row index, used for the annotated heatmap
                qpt_bin_good.append(h)         # column index, used for the annotated heatmap 
    
    # PLOT accumulator2 
    fig,ax = plt.subplots(figsize=(8,4))
    plt.imshow(accumulator2)
    plt.xlim(0, phi0_bins)
    plt.ylim(0, qinvpt_bins)

    plt.ylabel('q/'r"$P_t$"'')                                                                     
    plt.xlabel(''r"$Φ_0$"'') 
    #ax.set_ylabel(''r"$Ψ$"'', rotation=0, fontsize=20, labelpad=20)
    ax.yaxis.set_label_coords(-0.05,0.89) 

    #ax.set_xlabel(''r"$Θ_0$"'', fontsize=20)       
    #phi0 = '\N{GREEK CAPITAL LETTER PHI}\N{SUBSCRIPT ZERO}'                                                               
    ax.set_title("Accumulator (" + str(qinvpt_bins) + "x" + str(phi0_bins) + ") bins for " + str(len(rho)) + " (r, Φ) values" + "\n")
    #plt.title("Accumulator (" + str(qinvpt_bins) + "x" + str(phi0_bins) + ") bins for " + str(len(rho)) + " (" + str(phi0) + ",Ψ) values" + "\n")
    #plt.colorbar()                                                                                                  # to comment for saving a clean image
    #plt.axis('off')
    img_name = "Acc-" + str(road_num) + "r-" + str(fake_num) +  "fc-" + ".png"
    plt.savefig(image_dir+img_name,dpi=fig.dpi, transparent=True)                                                                                                                                                       
    fig.tight_layout()
    plt.show()
    plt.close(fig)                 
    
    #accumulatorAsAnnotatedHeatmap(accumulator2, phi0_bins_good, qpt_bin_good)

    return (accumulator2, count)


def accumulatorAsAnnotatedHeatmap(accumulator, indici, m_trovato):
    q1 = deltaBinPhi0*np.arange(phi0_bins) 
    deltaIndex = 200# 60
    i_median=int(statistics.median(indici))                                                             
    m_max = max(m_trovato) + 20
    i_min = -deltaIndex+i_median
    i_max = +deltaIndex+i_median
     
    q2 = q1[i_min:i_max]
    mini_accumulator = accumulator[0:m_max, i_min:i_max]
    fig, ax = plt.subplots()
    ax.imshow(mini_accumulator)                                                                
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # text annotations
    for i in range(m_max):
        for k in range(len(q2)):
            ax.text(k, i, mini_accumulator[i, k], ha="center", va="center", color="w")
    
    #ax.set_title("Portion of the Accumulator (" + str(m_max) + "x" + str(len(q2)) +") \n")              
    fig.tight_layout()
    accumulatorAnnotated_name = "accumulatorAnnotated_"  + ".png"
    plt.savefig(accumulatorAnnotated_name, dpi=fig.dpi, transparent=True)                                         # not good quality for the saved images.
    plt.axis('off')
    plt.show()
    plt.close()

# Research for value equal to 7, 8 in the accumulator 
def researchRoads(accumulator,  phi_road, invpt_road, road, extracted_roads, fakeORnot, road_number, fake_clusters):
    
    myDataFile1 = open(outFile, "w")
    phi0_original_road = []              # is the array containing the phi0 value of the original/real roads generated by the TV-gen.py program BUT recognized by the check
    qinvpt_original_road = []            # is the array containing the qinvPt value of the original/real roads generated by the TV-gen.py program
    found_roads = 0
    count2 = 0
    found_road_equal_to_original_ones = []
    total_clusters = len(phi_road)
    density = 0
    max_accum_density = n_layers*qinvpt_bins*phi0_bins

    for i in range(qinvpt_bins):
        for j in range(phi0_bins):   
            density += accumulator[i,j]                                                               
            if ( (accumulator[i,j] >= threshold)):                    
                # qinvPt_road and phi0_road must be reconverted before being compared
                qinvPt_road = i*deltaBinQinvpt + qinvpt_min    
                phi0_road = phi0_min+deltaBinPhi0*j  
                #print("With " + str(accumulator[i,j]) + ":")
                #print("phi0=" + str(phi0_road) + "\n" + "invpt="+ str(qinvPt_road) )
                myDataFile1.write("Found road with " + str(accumulator[i,j]) )
                myDataFile1.write(" phi0= " + str(phi0_road) + ", invpt="+ str(qinvPt_road) + "\n")    
                
                # Extracted AND REAL (generated) ROADs VALUES COMPARISON
                # We compare the (phi0_road,qinvPt_road) extracted road values with the ones in the inFile_01 file (phi_road, invpt_road parameters). 
                # A road is considered found if the difference between the 2 values is less than deltaBinPhi0 for phi0 and deltaBinQinvpt for q/Pt)
                for k in range(len(phi_road)):
                    
                    #print("phi0_road - phi_road[k]=",phi0_road - phi_road[k])
                    #print("qinvPt_road - invpt_road[k]=",qinvPt_road - invpt_road[k])

                    if ( abs(phi0_road - phi_road[k]) < deltaBinPhi0 and abs(qinvPt_road - invpt_road[k]) < deltaBinQinvpt ):
                        if (phi_road[k] not in phi0_original_road ) and (invpt_road[k] not in qinvpt_original_road):
                            found_roads += 1
                            #print("Road number", found_roads)
                            #print("phi0: " + str(abs(phi0_road - phi_road[k]))  + "< " + str(deltaBinPhi0 ))
                            #print("invpt: " + str(abs(qinvPt_road - invpt_road[k]))  + "< " + str(deltaBinQinvpt ))
                            if (fakeORnot[k] == "Y"):
                                found_road_equal_to_original_ones.append(road_number[k])
                                count2 +=1
                                
                            # per capire quali road presenti nel file di input vengono riconosciute; ci può essere rumore che viene riconosciuto come road buone
                            phi0_original_road.append(phi_road[k])      # delle road riconosciute sono i phi0 presenti nel file di partenza
                            qinvpt_original_road.append(invpt_road[k])  # delle road riconosciute sono gli invpt presenti nel file di partenza
    
    perc_density = density/max_accum_density
    
    percentage = len(found_road_equal_to_original_ones)/road
    plus_extracted_roads = extracted_roads - road

    # Print results
    print("Total clusters: ",total_clusters)
    print("Fake clusters:", fake_clusters)
    #print("seed =", seed)
    print("Real roads: " + str(road))
    cprint("Extracted roads (treshold >= " + str(threshold) +  "): " + str(extracted_roads) , color='green')      # admitted colors: red, cyan, blue, green, magenta
    cprint("Exceeded (extracted) roads: " + str(plus_extracted_roads), color = 'green')
    cprint("Found roads: " + str(found_roads) + " - of which " + str(count2) + " are original roads.")
    #cprint("Found roads: " + str(found_roads) + " - of which " + str(count2) + " are original roads: " + str(found_road_equal_to_original_ones), color='green')
    print("Success: " + str("{:.2f}".format(100*percentage)) + "%")
    print("Accumulator density: " + str("{:.2f}".format(perc_density*100)) + "%")

    myDataFile1.close()
    return (found_roads, percentage, perc_density)


def main():  

        start = datetime.datetime.now()

        roads = [5,10,20,40]                                            #np.arange(5, 41)  # [5,10,20,40]                  
        perc_fake = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]             #[0, 0.33, 0.5, 0.75, 0.88, 0.94] 
        
        # To create statistics output filename
        x = datetime.datetime.now().strftime("%b_%d-%H.%M")
        outStatsFile = "TVStats-" + str(qinvpt_bins) + "x" + str(phi0_bins) + "-" + str(grana_qinvpt) + "-" + str(grana_phi0) + "-" + str(x) + ".txt"
        
        f = open(outStatsFile, "w")
        f.write("#Total cluster, Fake cluster, Percentage fake, Real roads, Extracted roads, Exceeded roads, Found roads, Percentage success, Percentage accumulator density" + "\n")

        for road in roads:
            TV_Gen_r(road)
            shutil.copy(outFile1, outFile2)         # copy the generated TV-Gen_r file in outFile2 (where there will be appended the fake clusters)
            for perc in perc_fake:
                fc = int(perc*road*16/(1-perc))     # fake clusters input data percentage calculus
                f1 = TV_Gen_fc(fc)
                phi, rho, l, phi_road, invpt_road, fakeORnot, road_number = dataExtraction(f1)    
                accumulator, extracted_roads = accumulatorCreation(phi,rho,l, road, fc)  
                found_roads, percentage, perc_density = researchRoads(accumulator, phi_road, invpt_road, road, extracted_roads, fakeORnot, road_number,fc)
                exceeded_roads = extracted_roads - road
                f.write(str(fc+road*16) + "," + str(fc)  + "," + str(perc) + "," + str(road) + "," + str(extracted_roads) + "," + str(exceeded_roads) + "," + str(found_roads) + "," + str("{:.2f}".format(percentage)) + "," + str("{:.2f}".format(perc_density)) +"\n")
        f.close()

        end = datetime.datetime.now()
        print("Total time of processing:",end-start)

if (__name__ == "__main__"):
    suppress_qt_warnings()
    main()