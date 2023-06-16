import sys
import numpy as np
from matplotlib.pylab import  plt

arr = np.fromfile(sys.argv[1], dtype=np.float32)
print(arr.shape, 19*192*384, 2*192*384, 1*192*384, 1*768*1536)
if arr.shape[0] == 19*192*384:
    r = np.argmax(np.reshape(arr, [19,192,384]), axis=0)
elif arr.shape[0] == 2*192*384:
    r = np.argmax(np.reshape(arr, [2,192,384]), axis=0)
elif arr.shape[0] == 1*192*384:
    r = np.reshape(arr, [1,192,384])
elif arr.shape[0] == 1*768*1536:
    r = np.reshape(arr, [1,768,1536])[0]
elif arr.shape[0] == 3*768*1536:
    r = np.transpose(np.reshape(arr, [3,768,1536]) - arr.min(), (1, 2, 0))
print(r.min(), r.max(), r.shape)
plt.imshow(r)
plt.show()
