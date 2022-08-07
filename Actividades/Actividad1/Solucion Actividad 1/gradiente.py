import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm
#Vamos a plotear la función gradiente definida en la parte a)
#definimos los valores de x e y dentro de la región (0,pi)x(0,pi)
x = np.linspace(0,np.pi)
y = np.linspace(0,np.pi)
X,Y = np.meshgrid(x,y)
#Definimos f
f= np.cos(X)*np.cos(Y)*np.exp(-(X**2)/5)

#A mano se determino que el gradiente de f es:
dfx = -( np.sin(X) +(2/5)*x*np.cos(X))*np.cos(Y)*np.exp(-(X**2)/5)
dfy = np.sin(Y)*np.cos(X)*np.exp(-(X**2)/5)

fig = plt.figure()
# Creamos el plano 3D
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, f,cmap=cm.coolwarm)
plt.xlabel("Eje X")
plt.ylabel("Eje Y")
plt.title("Gráfico de la función f en la región (0,pi)x(0,pi)")

plt.show()



fig = plt.figure()
plt.quiver(X,Y,dfx,dfy)
plt.xlabel("df x")
plt.ylabel("df y")
plt.title("Gradiente de f en la región (0,pi)x(0,pi)")
plt.show()

