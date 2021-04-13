# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 16:45:52 2020

@author: SAKSHAM
"""


import numpy as np
import cv2
import pywt
from PIL import Image
import matplotlib.pyplot as plt

##############################################################################################################
# For calculating image energy of given image----------------------------------------------------------------

def calc_energy_map(img):
        energy = np.absolute(cv2.Scharr(img, -1, 1, 0)) + np.absolute(cv2.Scharr(img, -1, 0, 1))
    #    print(energy.shape)
        return energy
##############################################################################################################   
    


##############################################################################################################
# For calculating a vertu=ical seam to be removed for the img--------------------------------------------------
def calc_seam_vert(img):
     m,n = img.shape
     dp = np.zeros((m,n))
     en_map = calc_energy_map(img)    # using the energy map function
     for j in range(m):
      #   print(en_map.shape, m, n)
         dp[0,j] = en_map[0,j]        # Dynamic Programming for storing previous computations
     m, n = img.shape
     seam = np.zeros((m,n))
     for i in (np.arange(m-1) + 1):
         for j in np.arange(n):
             if j == 0:
                 dp[i,j] =  en_map[i,j] + min(dp[i-1,j], dp[i-1,j+1])  # DP step
                 x = dp[i,j] - en_map[i,j]
                 if x == dp[i-1,j]:
                     seam[i,j] = j
                 if x == dp[i-1,j+1]:
                     seam[i,j] = j +1  
                     
             if j == n-1:
                 dp[i,j] =  en_map[i,j] + min(dp[i-1,j-1],dp[i-1,j])
                 x = dp[i,j] - en_map[i,j]
                 if x == dp[i-1,j]:
                     seam[i,j] = j
                 if x == dp[i-1,j-1]:
                     seam[i,j] = j -1  
             if j != 0 and j != n-1:     
                 dp[i,j] = en_map[i,j] + min(dp[i-1,j-1], min(dp[i-1,j], dp[i-1,j+1]))
                 x = dp[i,j] - en_map[i,j]
                 if x == dp[i-1,j]:
                     seam[i,j] = j
                 if x == dp[i-1,j-1]:
                     seam[i,j] = j -1  
                 if x == dp[i-1,j+1]:
                     seam[i,j] = j +1             
     min_indx = np.argmin(dp[m-1,:])           # Minimum over all elemeents of first row
     sm = []
     sm += [min_indx]
     j = min_indx
     for i in np.arange(m-1, 0 , -1):
         j = int(j)
         sm += [seam[i,j]]
         j = seam[i,j]
         
     sm = np.flip(sm)
     return sm                                # Returnming the seam
 
##############################################################################################################   
    


##############################################################################################################
# For calculating a horizontal seam to be removed for the img--------------------------------------------------    
    
def calc_seam_hori(img):
      img2 = np.transpose(img)     # Transposing and calculating vetical seam
      sm = calc_seam_vert(img2)
      return sm


##############################################################################################################   
    


##############################################################################################################
# For removing a horizontal seam to be removed for the img--------------------------------------------------  
def remove_seam_hori(img):
    m,n = img.shape
    sm = calc_seam_hori(img)
    nw_img = np.zeros((m-1, n))    # initiaixzins the new image array
    for j in np.arange(n):
        col = img[:,j]
        col = np.delete(col, sm[j])    # deleting the corresponding element from seam
       # print(j, col.shape, (nw_img[:,j]).shape)
        col = np.array(col)
        nw_img[:,j] = col
    return nw_img    

##############################################################################################################   
    


##############################################################################################################
# For removing a vertical seam to be removed for the img--------------------------------------------------  

        

def remove_seam_vert(img):
    m,n = img.shape
    sm = calc_seam_vert(img)
    nw_img = np.zeros((m, n-1))
    for i in range(m):
        col = img[i,:]
        col = np.delete(col, sm[i])
        col = np.array(col)
        nw_img[i,:] = col
    return nw_img       

##############################################################################################################   
    


##############################################################################################################
# For resizing a given image to a particular size--------------------------------------------------  
def resize_img(img, _m, _n):
    m, n = img.shape
    for i in range(m - _m):
        img = remove_seam_hori(img)    # calling horizontal seams
    for i in range(n - _n):
        img = remove_seam_hori(img)      # calling verctical seams
    return img  

##############################################################################################################   
    


##############################################################################################################
# For calculating total energy of the givan image--------------------------------------------------  
def img_energy(img):
    en = calc_energy_map(img)   # calling energy map function
    return np.sum(en)
##############################################################################################################   
    


##############################################################################################################
# For uniform resizing and calculating energy for each size--------------------------------------------------          


def util(img):
    en = []
    for i in range(200):
        img  = remove_seam_vert(img)    # calling horizontal seam
        img = remove_seam_hori(img)    # calling vertical seam
        en += [img_energy(img)]        # energy
    return en
    
##############################################################################################################        

im = Image.open('eagle.tif')    # importing the image
img = np.array(im)              # convert to array
arr = util(img)

## ##########################    Plotting  energy vs dim
plt.plot(np.arange(256, 56, -1), arr)    # Plot
plt.ylabel('Energy Retained')
plt.xlabel('Dimension of Reduced Image')
plt.title('Plot of Energy vs Dimensions retained')
plt.show()
#####################


#### Compression 

for i in range(50):
    img  = remove_seam_vert(img)
    img = remove_seam_hori(img)
    
img =  img.astype(int)
img  = np.array(img, dtype=np.uint8)
cv2.imshow('image',img)
cv2.waitKey(0)

print(img_energy(img))  

######################### END OF CODE #########################################################################

                   
                   
     