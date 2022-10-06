import numpy as np
import matplotlib.pyplot as plt

infile = './TNG50-1_halpha.npy'

data=np.load(infile)

plt.imshow(data)
plt.show()

plt.imsave(infile[:-4]+'.jpeg', data)
