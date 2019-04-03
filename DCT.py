import scipy.io
from scipy import signal
import numpy as np
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from numpy import r_
import matplotlib.pyplot as plt
from math import log10
import matplotlib.pylab as pylab

def cal_snr(image_a, image_b):
    # calculate mean square error between two images
    var_a = np.var(image_a.astype(float))
    var_b = np.var(image_b.astype(float) - image_a.astype(float))
    snr = 10 * log10(var_a / var_b)
    return snr

def dct2(a):
    return scipy.fftpack.dct(scipy.fftpack.dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def idct2(a):
    return scipy.fftpack.idct(scipy.fftpack.idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


im = scipy.io.loadmat('Peppers.mat')['peppers']

# plt.imshow(im, cmap='gray', vmax=255, vmin=200)

imsize = im.shape
dct = np.zeros(imsize)

block_size = 8

# Do 8x8 DCT on image (in-place)
for i in r_[:imsize[0]:block_size]:
    for j in r_[:imsize[1]:block_size]:
        dct[i:(i + block_size), j:(j + block_size)] = dct2(im[i:(i + block_size), j:(j + block_size)])

# pos = 128
# # Extract a block from image
# plt.figure()
# plt.imshow(im[pos:pos + block_size, pos:pos + block_size], cmap='gray')
# plt.title("An 8x8 Image block")
#
# # Display the dct of that block
# plt.figure()
# plt.imshow(dct[pos:pos + block_size, pos:pos + block_size], cmap='gray', vmax=np.max(dct) * 0.01, vmin=0, extent=[0, pi, pi, 0])
# plt.title("An 8x8 DCT block")
#
# plt.figure()
# # plt.imshow(dct, cmap='gray')
#
# plt.imshow(dct, cmap='gray', vmax=np.max(dct) * 0.01, vmin=0)
# plt.title("8x8 DCTs of the image")

# Threshold
thresh = 0.01
dct_thresh = dct * (abs(dct) > (thresh * np.max(dct)))

# plt.figure()
# plt.imshow(dct_thresh, cmap='gray')
#
# # plt.imshow(dct_thresh, cmap='gray', vmax=np.max(dct) * 0.01, vmin=0)
# plt.title("Thresholded 8x8 DCTs of the image")

percent_nonzeros = np.sum(dct_thresh != 0.0) / (imsize[0] * imsize[1] * 1.0)

print("Keeping only %f%% of the DCT coefficients" % (percent_nonzeros * 100.0))

im_dct = np.zeros(imsize)

for i in r_[:imsize[0]:block_size]:
    for j in r_[:imsize[1]:block_size]:
        im_dct[i:(i + block_size), j:(j + block_size)] = idct2(dct_thresh[i:(i + block_size), j:(j + block_size)])

# plt.figure()
# plt.imshow(np.hstack((im, im_dct)), cmap='gray')
# plt.title("Comparison between original and DCT compressed images")

# snr
snr = cal_snr(im,im_dct)
print('SNR:'+str(snr))

# plt.imsave('5-16.png',im_dct,cmap='gray')
# Display the dct of that block
fig1, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(dct, cmap='gray', vmax=np.max(dct) * 0.01, vmin=0)
axarr[0, 0].set_title('16x16 DCTs of the Image')

axarr[0, 1].imshow(dct_thresh, cmap='gray', vmax=np.max(dct) * 0.01, vmin=0)
axarr[0, 1].set_title('Thresholded 16x16 DCTs of the Image')

axarr[1, 0].imshow(im, cmap='gray')
axarr[1, 0].set_title('Original Image')

axarr[1, 1].imshow(im_dct, cmap='gray')
axarr[1, 1].set_title('16x16 DCT Compressed Image')
plt.show()

