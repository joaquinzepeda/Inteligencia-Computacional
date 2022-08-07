import numpy as np
import matplotlib.pyplot as plt
import random

# media y desvio estandar
mu1, sigma1 = -1, 1
datos1 = np.random.normal(mu1, sigma1, 100) #creando muestra de datos <class 'numpy.ndarray'> (1000,)
#salida1 = np.zeros(100, dtype=int)

mu2, sigma2 = 1, 1
datos2 = np.random.normal(mu2, sigma2, 100) #creando muestra de datos <class 'numpy.ndarray'> (100,)

#salida2 = np.ones(100, dtype=int)


datos=[]
for dato in datos1:
    datos.append((dato,0))

for dato in datos2:
    datos.append((dato,1))

random.shuffle(datos)
#print(datos)

count, bins, ignored = plt.hist(datos1, 30, density=True)
plt.plot(bins, 1/(sigma1 * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu1)**2 / (2 * sigma1**2) ),
         linewidth=2, color='cyan')
count, bins, ignored = plt.hist(datos2, 30, density=True)
plt.plot(bins, 1/(sigma2 * np.sqrt(2 * np.pi)) *
               np.exp( - (bins - mu2)**2 / (2 * sigma2**2) ),
         linewidth=2, color='r')
plt.title("Dataset aleatorio de distribución normal")
plt.show()


'''
bins = np.linspace(-10, 10, 100)

plt.hist([datos1, datos2], bins, label=['a', 'b'])
plt.legend(loc='upper left')
plt.show()
'''


def salida(s):
    #print("valor de s",s)
    y = 1/(1+np.exp(s))
    return y
def entropia_cruzada(t,y):
    return -t*np.ln(y)-(1-t)*np.ln(1-y)

b0 = 0.4 #b al comienzo es igual a 0.4
w0=-0.1 #w al comienzo es igual a -0.1
def regla_aprendizaje(datos, learning_rate, n_iteraciones=1):  
    global b0,w0
    b0 = 0.4 #b al comienzo es igual a 0.4
    w0=-0.1 #w al comienzo es igual a -0.1
    #t=target
    n=0
    arreglo_w=[-0.1]
    arreglo_b=[0.4]
    entropia=[]
    while n_iteraciones>0:
        suma_w0=0
        suma_b0=0
        for x,t in datos:
            #print(x,t)
            argumento= -(w0*x+b0)
            #print("argumento n°"+str(n_iteraciones),argumento)
            y=salida(argumento)
            if y==1:
                print("error, division por cero")
            #print("y",y)
            suma_w0+=learning_rate*(-t*y*np.exp(argumento)*x + (1-t)*y**2*np.exp(argumento)*x/(1-y))
            suma_b0+=learning_rate*(-t*y*np.exp(argumento) + (1-t)*y**2*np.exp(argumento)/(1-y))

        entropia.append(-t*np.log(y)-(1-t)*np.log(1-y)) #guardo el costo de cada iteracion

        promedio_w0 = suma_w0/len(datos)
        promedio_b0 = suma_b0/len(datos)
        #print("promedio w0",promedio_w0)
        w0 = w0 - promedio_w0
        b0 = b0 - promedio_b0
        if n%20==0:
            print(f"iteración N°{n} de w0, b0",w0,b0)
        arreglo_w.append(w0)
        arreglo_b.append(b0)
        n_iteraciones-=1
        n+=1

    return n,arreglo_w,arreglo_b,entropia


#obtenemos los valores de los pesos y el bias arreglados por el algoritmo
tiempo, arreglo_w,arreglo_b,entropia = regla_aprendizaje(datos,0.1,100)

print("Despues del entrenamineto, los valores obtenidos de w0,b0 son respectivamente: ",w0,b0,"\n")



'''
print("procedo a hacer una revisión de los datos obtenidos (ojo que es posible encontrar errores y/o outlayers)")
for i in range(5):
    x,t=datos[i]
    print(f"Prueba N°{i} dato,target",x,t)
    argumento= -(w0*x+b0)
    #print("argumento n°"+str(n_iteraciones),argumento)
    y=salida(argumento)
    print("resultado",y)
'''

base = np.linspace(-10,10,1000)
y = salida(-(w0*base+b0))
plt.plot(base,y)
plt.title("Modelo de la función")
plt.xlabel("Input")
plt.ylabel("Output")
plt.show()

base = np.linspace(-10,10,100)
plt.scatter(base,entropia)
plt.title("Curva de aprendizaje ")
plt.xlabel("Iteraciones")
plt.ylabel("Costo de entropia cruzada")
plt.show()

fig = plt.figure()
ax1 = fig.add_subplot(111)
base = np.linspace(-10,10,101)
ax1.scatter(base, arreglo_w, s=10, c='b', marker="s", label='b bias')
ax1.scatter(base,arreglo_b, s=10, c='r', marker="o", label='w pesos')
plt.title("Evolución de los parametros ")
plt.xlabel("Iteraciones")
plt.ylabel("Valor")
plt.legend(loc='upper left');
plt.show()

print("¿Por qué w no se va a infinito? Imagine qué efecto tendría esto sobre la función de costos en el caso de una muestra mal clasificada.\n")
print("w no se va a infinito porque el gradiente toma valores acotados, además de esto, la tasa de aprendizaje regula que tanto cambia el valor de w y esta regula tambien que no tienda a infinito.\n")