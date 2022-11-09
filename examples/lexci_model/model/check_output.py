import numpy as np
import matplotlib.pyplot as plt

infile = 'examples/lexci_model/build/lexci_model_output.dat'

data=np.loadtxt(infile)
data = np.reshape(data, (256,256))

plt.imshow(data)
plt.show()

plt.imsave(infile[:-4]+'.jpeg', data, cmap='Greys')
