"""
Create focus metric files for every image using 5 metrics (lapSD,dark,sobSD,damp,Gabor)

v9 01.25.21  Removed unused code
v8 12.12.21  Added Gabor 0 * 90
V4 09.19.21  Posted in github
V2 08.13.21  Fixed rollover bug in focus that affects dark calculation
             Dark5Score is mean of bottom 5% of dark pixel values
V1 06.13.21

Tom Zimmerman, IBM Research-Almaden, Center for Cellular Construction
This material is based upon work supported by the NSF under Grant No. DBI-1548297.  
Disclaimer:  Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation
"""
import numpy as np
import cv2
from os import listdir 
from os.path import isfile, join
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import reconstruction as reco

#################### USER DEFINED DIRECTORIES #######################
rawImageDir=r'~\rawImageZ\\'        # where to find raw images to reconstruction
csvDir=r'~\focusScores\\'           # where to place focus scores

############ CONSTANTS #####################
header='z,lapSD,dark,sobSD,damp,Gabor'
DEBUG=0                     # set to see output images
IMAGE_NUMBER=0; IMAGE_Z=1;  # image naming convention
DISPLAY_REZ=(480,480)       # 960,540
FOCUS_METHODS=5             # lapSD,dark,sobSD,damp,Gabor0,Gabor90
Z_INDEX=0                   # location in focus csv spreadsheet
DELTA_FLAG=0                # 1= remove background for reco
HANN_FLAG=0                 # 1=use Hann window in reco
DAMP=4                      # location in focus vector
zStep = 20                  # how many Z values to step (um)
dxy   = 1.4e-6              # imager pixel size in meters.
zScale=1e-6                 # convert z units to microns 
minZ = 20                   # minimum possible Z value (um)
maxZ = 3000                 # maximum possible Z value (um)  
wvlen = 660.0e-9            # wavelength of laser. Blue is 405 nm. Red is 650. [Nayak is 660 nm]
MASK_DAMP=2                 # first few samples of DAMP too high so correct
ampIM_last=np.zeros((1,1),dtype='uint8')
complexIM_last=np.zeros((1,1),dtype='uint8')

################# FUNCTIONS ###################
def focus(im, z):
    global ampIM_last,complexIM_last

    (xRez,yRez) = im.shape
    
    # Reconstruct the image at this Z.
    (ampIM, complexIM)=reco.recoFrame(im, z,DELTA_FLAG,HANN_FLAG)
    if DEBUG:
        cv2.imshow('recoIM',cv2.resize(ampIM,DISPLAY_REZ))
        cv2.waitKey(20)
    
    # Find min change in complex sum
    if complexIM_last.shape!=complexIM.shape:
        complexIM_last=complexIM
        damp=999999 # make it really large because the smallest value indicates focus
    else:
        damp = np.sum(np.abs (np.abs(complexIM[:]) - np.abs(complexIM_last[:]) ) ** 2)
        complexIM_last=complexIM
     
    # Dark bottom 2%  
    (yRez,xRez) = im.shape
    area=xRez*yRez
    bottom2percent = int(0.02 * area)
    darkSort=np.sort(ampIM,axis=None)
    dark=np.sum(darkSort[0:bottom2percent])/bottom2percent
    
    # Determining focus score by mean of Laplacian 
    kernel_size = 3; ddepth = cv2.CV_64F
    lap = cv2.Laplacian(ampIM, ddepth, ksize=kernel_size)
    lapMSD = cv2.meanStdDev(lap)
    lapSD = lapMSD[1][0][0]
    
    # Determining focus score by mean of Sobel 
    sobel = np.abs(cv2.Sobel(ampIM,ddepth,1,1,ksize=kernel_size))
    sobMSD=cv2.meanStdDev(sobel)
    sobSD=sobMSD[1][0][0] # standard deviation

    # Determine Gabor filter result
    g0 = ndi.convolve(ampIM, gkernel0, mode='wrap')
    g90 = ndi.convolve(ampIM, gkernel90, mode='wrap')
    #g = np.multiply(g0,g90)
    gv0 = g0.var()
    gv90 = g90.var()
    gabor=np.multiply(gv0,gv90)
    return(lapSD,dark,sobSD,damp,gabor)
    

def norm(data):
    return (data - np.amin(data,axis=0)) / (np.amax(data,axis=0) - np.amin(data,axis=0))

######################  MAIN  #####################
print('Start')

# prepare gabor filter kernel
theta=0
angle = theta / 4. * np.pi
sigma=1         # 4,9,12
frequency=0.35  # 0.125, 0.25,0.35,0.50
gkernel0 = np.real(gabor_kernel(frequency, theta=angle,sigma_x=sigma, sigma_y=sigma))
theta=90
angle = theta / 4. * np.pi
gkernel90 = np.real(gabor_kernel(frequency, theta=angle,sigma_x=sigma, sigma_y=sigma))

# find files in directory
files = [f for f in listdir(rawImageDir) if isfile(join(rawImageDir, f))]
   
for f in files:         # get focus values for all raw images
    r=f.split('_')
    imageNumber=r[IMAGE_NUMBER]
    trueZ=int(r[IMAGE_Z])
    print('imageNumber',imageNumber,'trueZ',trueZ)

    # get image
    rawIM = cv2.imread(rawImageDir+f, 0)            # read the image as grayscale.

    # calculate focus metrics over z stack range
    maxStep=int((maxZ-minZ)/zStep)
    focusScore=np.zeros((maxStep,1+FOCUS_METHODS))  # z,focus metrics...
    index=0
    for z in range(minZ, maxZ, zStep):
        focusScore[index,0]=z
        focusScore[index,1:]=focus(rawIM, z)
        index+=1

    # since first value of damp is incorrect, for it to be equal to the second value
    focusScore[0:MASK_DAMP,DAMP]=focusScore[MASK_DAMP,DAMP]

    imageNumberCol=focusScore[:,Z_INDEX] # save z values so they won't be messed up by normalization
    fileName=str(imageNumber)+'_'+str(trueZ)+'_.csv'
    np.savetxt(csvDir+fileName,focusScore,delimiter=',',header=header,fmt='%f')
    print('Saved file',fileName)
  
