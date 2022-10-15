from sklearn.datasets import load_sample_images
from matplotlib import pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd

dataset = load_sample_images()
print(dataset.keys())
images = dataset['images']
window_name = 'image'
img = np.flip(images[1], axis=-1)
img2 = images[1]
cv.imshow(window_name , img)

#print(dataset[0])
b, g, r = cv.split(img)
a = [r.size,g.size,b.size]
print(a)

cv.waitKey(0)
cv.destroyAllWindows()