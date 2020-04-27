#%%
import cv2
from matplotlib import pyplot as plt
import mclahe


img = cv2.imread('filter_test/2000-09-10_RSAT_20_3.png',  cv2.IMREAD_GRAYSCALE)

plt.imshow(img, cmap='gray', vmin = 0, vmax = 255)
plt.show()

#%%
img_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25,25)).apply(img) # CLAHE adaptive contrast enhancement
plt.imshow(img_CLAHE, cmap='gray', vmin = 0, vmax = 255)
plt.show()
#%%
img_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25,25)).apply(img) # CLAHE adaptive contrast enhancement
plt.imshow(img_CLAHE, cmap='gray', vmin = 0, vmax = 255)
plt.show()

#%%
imfile = 'NLMeans_default'
img = cv2.imread('filter_test/' + imfile + '.png',  cv2.IMREAD_GRAYSCALE)
img_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(25,25)).apply(img) # CLAHE adaptive contrast enhancement
cv2.imwrite('filter_test/clahe_'+ imfile + '.png', img_CLAHE)

