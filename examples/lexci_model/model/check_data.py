import numpy as np
import matplotlib.pyplot as plt

infile = 'examples/lexci_model/model/TNG50-1_halpha.npy'

data=np.load(infile)

plt.imshow(data)
plt.show()

plt.imsave(infile[:-4]+'.jpeg', data, cmap='Greys')
