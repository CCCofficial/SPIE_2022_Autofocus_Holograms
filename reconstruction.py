'''
Perform holographic reconstruction with angular spectrum method

v3 01.25.22 removed unused code
v3 12.12.21 added Hann window feature
v2 11.07.21 cropped images has even value coordinates

Tom Zimmerman, IBM Research-Almaden, Center for Cellular Construction
This material is based upon work supported by the NSF under Grant No. DBI-1548297.  
Disclaimer:  Any opinions, findings and conclusions or recommendations expressed in this material are those of the authors and do not necessarily reflect the views of the National Science Foundation
'''
import numpy as np
import cv2

dxy   = 1.4e-6 # imager pixel size in meters.
wvlen = 650.0e-9 # Red 
zScale=1e-6 # convert z units to microns

def recoFrame(cropIM, z,DELTA_FLAG,HANN_FLAG): # DELTA_FLAG=1 means remove DC component, HANN_FLAG means apply Hann Window before reco
    (r,c)=cropIM.shape
    cropIM=cropIM[0:r&0xFFFE,0:c&0xFFFE] # make coordinates even for FFT
    
    if DELTA_FLAG:
        cropIM=deltaImage(cropIM)
    if HANN_FLAG:
        (r,c)=cropIM.shape
        hannWindow=cv2.createHanningWindow((c,r), cv2.CV_64FC1)
        cropIM=cropIM*hannWindow
        
    complex = propagate(np.sqrt(cropIM), wvlen, z*zScale, dxy)	 #calculate wavefront at z
    amp = np.abs(complex)**2          # output is the complex field, still need to compute intensity via abs(res)**2
    amp = np.clip(amp,0,255)        # prevent rollover when converting to 8 bit
    ampInt = amp.astype('uint8')
    return(ampInt, complex)

def propagate(input_img, wvlen, zdist, dxy):
    M, N = input_img.shape # get image size, rows M, columns N, they must be even numbers!

    # prepare grid in frequency space with origin at 0,0
    _x1 = np.arange(0,N/2)
    _x2 = np.arange(N/2,0,-1)
    _y1 = np.arange(0,M/2)
    _y2 = np.arange(M/2,0,-1)
    _x  = np.concatenate([_x1, _x2])
    _y  = np.concatenate([_y1, _y2])
    x, y  = np.meshgrid(_x, _y)
    kx,ky = x / (dxy * N), y / (dxy * M)
    kxy2  = (kx * kx) + (ky * ky)

    # compute FT at z=0
    E0 = np.fft.fft2(np.fft.fftshift(input_img))

    # compute phase aberration
    _ph_abbr   = np.exp(-1j * np.pi * wvlen * zdist * kxy2)
    output_img = np.fft.ifftshift(np.fft.ifft2(E0 * _ph_abbr))
    return output_img

def deltaImage(grayIM):
    # subract off blurred image to get rid of DC component
    # return floating point version of image, offset for all positive values
    blur=151
    blurIM=cv2.medianBlur(grayIM,blur)  # blur image to get avg brightness

    g=grayIM.astype(float)
    b=blurIM.astype(float)
    delta=g-b
    d0min=np.amin(delta)
    d0max=np.amax(delta)
    if d0min<0:
        delta+=abs(d0min)
    delta *= 255.0/delta.max() 
    delta=np.clip(delta,0,255)
    delta=delta.astype('uint8')
    return(delta)
