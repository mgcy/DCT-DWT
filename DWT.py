import numpy as np
import pywt
from math import log10
import matplotlib.pyplot as plt
import scipy.io


def cal_snr(image_a, image_b):
    # calculate mean square error between two images
    var_a = np.var(image_a.astype(float))
    var_b = np.var(image_b.astype(float) - image_a.astype(float))
    snr = 10 * log10(var_a / var_b)
    return snr


im = scipy.io.loadmat('Peppers.mat')['peppers']
##########################################################################
'''
# 2 level
cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = pywt.wavedec2(
    data=im,
    wavelet='db3',
    level=2)

bin_num = 15
# figures before thresholding
fig1, axarr1 = plt.subplots(2, 3)
axarr1[0, 0].imshow(cH2, cmap='gray')
axarr1[0, 0].set_title('Horizontal')
axarr1[0, 1].imshow(cV2, cmap='gray')
axarr1[0, 1].set_title('Vertical')
axarr1[0, 2].imshow(cD2, cmap='gray')
axarr1[0, 2].set_title('Diagonal')

fig2, axarr2 = plt.subplots(2, 3)
axarr2[0, 0].hist(cH2.reshape(cH2.shape[0] * cH2.shape[1]), bins=bin_num)
axarr2[0, 0].set_title('Horizontal')
axarr2[0, 1].hist(cV2.reshape(cV2.shape[0] * cV2.shape[1]), bins=bin_num)
axarr2[0, 1].set_title('Vertical')
axarr2[0, 2].hist(cD2.reshape(cD2.shape[0] * cD2.shape[1]), bins=bin_num)
axarr2[0, 2].set_title('Diagonal')
# compression
thresh = 0.1
cH2 = cH2 * (abs(cH2) > (thresh * np.max(cH2)))
cV2 = cV2 * (abs(cV2) > (thresh * np.max(cV2)))
cD2 = cD2 * (abs(cD2) > (thresh * np.max(cD2)))
cH1 = cH1 * (abs(cH1) > (thresh * np.max(cH1)))
cV1 = cV1 * (abs(cV1) > (thresh * np.max(cV1)))
cD1 = cD1 * (abs(cD1) > (thresh * np.max(cD1)))

nonzero = np.sum(cH2 != 0.0) + np.sum(cV2 != 0.0) + np.sum(cD2 != 0.0) + np.sum(cH1 != 0.0) + np.sum(
    cV1 != 0.0) + np.sum(cD1 != 0.0)

percent_nonzeros = (nonzero + 128 * 128) / (512 * 512)

print("Keeping only %f%% of the DCT coefficients" % (percent_nonzeros * 100.0))

# figures after thresholding
axarr1[1, 0].imshow(cH2, cmap='gray')
axarr1[1, 1].imshow(cV2, cmap='gray')
axarr1[1, 2].imshow(cD2, cmap='gray')

axarr2[1, 0].hist(cH2.reshape(cH2.shape[0] * cH2.shape[1]), bins=bin_num)
axarr2[1, 1].hist(cV2.reshape(cV2.shape[0] * cV2.shape[1]), bins=bin_num)
axarr2[1, 2].hist(cD2.reshape(cD2.shape[0] * cD2.shape[1]), bins=bin_num)

# reconstruction
im_rec = pywt.waverec2(coeffs=[cA2, (cH2, cV2, cD2), (cH1, cV1, cD1)],
                       wavelet='db3')

# Display the dct of that block
fig4, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(cA2, cmap='gray')
axarr[0, 0].set_title('cA')

axarr[0, 1].imshow(cH2, cmap='gray')
axarr[0, 1].set_title('cH')

axarr[1, 0].imshow(cV2, cmap='gray')
axarr[1, 0].set_title('cV')

axarr[1, 1].imshow(cD2, cmap='gray')
axarr[1, 1].set_title('cD')
# plt.show()

# snr
snr = cal_snr(im, im_rec)
print('SNR:' + str(snr))

fig3, axarr = plt.subplots(1, 2)
axarr[0].imshow(im, cmap='gray')
axarr[0].set_title('Original Image')
axarr[1].imshow(im_rec, cmap='gray')
axarr[1].set_title('Reconstructed Image')
plt.show()
'''
#########################################################################
# 2 level
cA2, (cH2, cV2, cD2), (cH1, cV1, cD1), (cH0, cV0, cD0) = pywt.wavedec2(
    data=im,
    wavelet='db3',
    level=3)

bin_num = 15
# figures before thresholding
fig1, axarr1 = plt.subplots(2, 3)
axarr1[0, 0].imshow(cH2, cmap='gray')
axarr1[0, 0].set_title('Horizontal')
axarr1[0, 1].imshow(cV2, cmap='gray')
axarr1[0, 1].set_title('Vertical')
axarr1[0, 2].imshow(cD2, cmap='gray')
axarr1[0, 2].set_title('Diagonal')

fig2, axarr2 = plt.subplots(2, 3)
axarr2[0, 0].hist(cH2.reshape(cH2.shape[0] * cH2.shape[1]), bins=bin_num)
axarr2[0, 0].set_title('Horizontal')
axarr2[0, 1].hist(cV2.reshape(cV2.shape[0] * cV2.shape[1]), bins=bin_num)
axarr2[0, 1].set_title('Vertical')
axarr2[0, 2].hist(cD2.reshape(cD2.shape[0] * cD2.shape[1]), bins=bin_num)
axarr2[0, 2].set_title('Diagonal')
# compression
thresh = 0.3
print('Threshold: '+str(thresh) )
cH2 = cH2 * (abs(cH2) > (thresh * np.max(cH2)))
cV2 = cV2 * (abs(cV2) > (thresh * np.max(cV2)))
cD2 = cD2 * (abs(cD2) > (thresh * np.max(cD2)))
cH1 = cH1 * (abs(cH1) > (thresh * np.max(cH1)))
cV1 = cV1 * (abs(cV1) > (thresh * np.max(cV1)))
cD1 = cD1 * (abs(cD1) > (thresh * np.max(cD1)))
cH0 = cH0 * (abs(cH0) > (thresh * np.max(cH0)))
cV0 = cV0 * (abs(cV0) > (thresh * np.max(cV0)))
cD0 = cD0 * (abs(cD0) > (thresh * np.max(cD0)))

nonzero = np.sum(cH2 != 0.0) + np.sum(cV2 != 0.0) + np.sum(cD2 != 0.0) + np.sum(cH1 != 0.0) + np.sum(
    cV1 != 0.0) + np.sum(cD1 != 0.0) + np.sum(cH0 != 0.0) + np.sum(
    cV0 != 0.0) + np.sum(cD0 != 0.0)

percent_nonzeros = (nonzero + 64 * 64) / (512 * 512)

print("Keeping only %f%% of the DCT coefficients" % (percent_nonzeros * 100.0))

# figures after thresholding
axarr1[1, 0].imshow(cH2, cmap='gray')
axarr1[1, 1].imshow(cV2, cmap='gray')
axarr1[1, 2].imshow(cD2, cmap='gray')

axarr2[1, 0].hist(cH2.reshape(cH2.shape[0] * cH2.shape[1]), bins=bin_num)
axarr2[1, 1].hist(cV2.reshape(cV2.shape[0] * cV2.shape[1]), bins=bin_num)
axarr2[1, 2].hist(cD2.reshape(cD2.shape[0] * cD2.shape[1]), bins=bin_num)

# reconstruction
im_rec = pywt.waverec2(coeffs=[cA2, (cH2, cV2, cD2), (cH1, cV1, cD1),(cH0, cV0, cD0)],
                       wavelet='db3')

# Display the dct of that block
fig4, axarr = plt.subplots(2, 2)
axarr[0, 0].imshow(cA2, cmap='gray')
axarr[0, 0].set_title('cA')

axarr[0, 1].imshow(cH2, cmap='gray')
axarr[0, 1].set_title('cH')

axarr[1, 0].imshow(cV2, cmap='gray')
axarr[1, 0].set_title('cV')

axarr[1, 1].imshow(cD2, cmap='gray')
axarr[1, 1].set_title('cD')
# plt.show()

# snr
snr = cal_snr(im, im_rec)
print('SNR:' + str(snr))

fig3, axarr = plt.subplots(1, 2)
axarr[0].imshow(im, cmap='gray')
axarr[0].set_title('Original Image')
axarr[1].imshow(im_rec, cmap='gray')
axarr[1].set_title('Reconstructed Image')
plt.show()
