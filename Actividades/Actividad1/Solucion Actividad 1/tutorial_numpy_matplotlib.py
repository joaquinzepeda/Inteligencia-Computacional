import numpy as np
import matplotlib.pyplot as plt
import time
import itertools

a = np.array([1,2,3])
print("arreglo a partir de una lista",a)

b=np.arange(1,10)
print(b)
c=np.zeros((2,4)) #arreglos de 2 por 4
print(c)


#RESHAPE CAMBIA LA FORMA DE UN ARREGLO
print(b)
a = np.arange(6).reshape((3, 2))
print(a)
import matplotlib.cm as cm 
