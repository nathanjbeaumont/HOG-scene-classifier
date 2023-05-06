from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from PIL import Image

from skimage.feature import hog
import numpy as np

database_path = Path(__file__).parents[1] / 'archive'
hog_list = []
label_list = []
for image_path in database_path.glob("**/*.jpg"):
    image = Image.open(image_path)
    image = np.asarray(image)
    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(16, 16),
                        cells_per_block=(2, 2), visualize=True, channel_axis=-1)
    hog_list.append(fd)
    label_list.append(image_path.parent.name)

hog_matrix = np.stack(hog_list, axis=0)
np.save(database_path / 'hog_matrix.npy', hog_matrix)
with open(database_path / 'labels.pkl', "wb") as fp:
    pickle.dump(label_list, fp)

# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# ax1.axis('off')
# ax1.imshow(image, cmap=plt.cm.gray)
# ax1.set_title('Input image')

# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# ax2.axis('off')
# ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# plt.show()
