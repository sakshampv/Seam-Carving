# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 02:01:51 2020

@author: SAKSHAM
"""

import numpy as np
import matplotlib.pyplot as plt
import pywt
import pywt.data
from PIL import Image
import cv2


##############################################################################################################
# For calculating image energy of given image----------------------------------------------------------------

def calc_energy_map(img):
        energy = np.absolute(cv2.Scharr(img, -1, 1, 0)) + np.absolute(cv2.Scharr(img, -1, 0, 1))
    #    print(energy.shape)
        return energy
##############################################################################################################   
        
##############################################################################################################
# For calculating total energy of the givan image--------------------------------------------------  
def img_energy(img):
    en = calc_energy_map(img)   # calling energy map function
    return np.sum(en)
##############################################################################################################   
    


############################################################################################################## 
# Load image
im = Image.open('eagle.tif')    # importing the image
img = np.array(im)              # convert to array

print(img_energy(img))
############################################################################################################## 



##############################################################################################################     
# L1
# Wavelet transform of image, and plot approximation and details
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
coeffs2 = pywt.dwt2(img, 'bior1.3')
LL, (LH, HL, HH) = coeffs2
fig = plt.figure(figsize=(12, 3))
for i, a in enumerate([LL, LH, HL, HH]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)   # Plotting results
    ax.set_title(titles[i], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])

fig.tight_layout()
plt.show()
############################################################################################################## 

############################################################################################################## 
# Inverse  transform for getting the image back
inverse_transform = pywt.idwt2((LL, (LH, HL, HH)), 'bior1.3')
inverse_transform =  inverse_transform.astype(int)
inverse_transform  = np.array(inverse_transform, dtype=np.uint8)
print(img_energy(inverse_transform))
cv2.imshow('image',inverse_transform)
cv2.waitKey(0)

############################################################################################################## 



############################################################################################################## 
#level 2

coeffs22 = pywt.dwt2(LL, 'bior1.3')      # Performing dwt to LL

inverse_transform2 = pywt.idwt2(coeffs22, 'bior1.3')    # Applying the inverse transform
inverse_transform2 =  inverse_transform2.astype(int)     # Converting to integer array
inverse_transform2  = np.array(inverse_transform2, dtype=np.uint8)
print(img_energy(inverse_transform2))
cv2.imshow('image',inverse_transform2)
cv2.waitKey(0)
############################################################################################################## 


