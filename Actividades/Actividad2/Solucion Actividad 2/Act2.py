import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Model


class LinearModelV2(Model):
    #inicializa el modelo
    def __init__(self,A=1,B=1,C=1,D=0,E=0,P=3.879):
        #llamamos a la clase super, que en este caso es Model
        super(LinearModelV2, self).__init__() 
        self.A = tf.Variable(A, dtype=tf.float32)
        self.B = tf.Variable(B, dtype=tf.float32)
        self.C = tf.Variable(C, dtype=tf.float32)
        self.D = tf.Variable(D, dtype=tf.float32)
        self.E = tf.Variable(E, dtype=tf.float32)
        self.P=tf.Variable(P, dtype=tf.float32)
        self.parameters = [self.A, self.B,self.C,self.D,self.E]
    
    #nos permite invocar el modelo, lo deja todo por defecto con 32 bits
    def call(self, t):
        P=self.P
        t=tf.cast(t, dtype=tf.float32)
        [A,B,C,D,E]=self.parameters
        
        y = A*tf.math.cos(2*np.pi*1/P*t) + B*tf.math.sin(2*np.pi*1/P*t) + C*tf.math.cos(2*np.pi*2/P*t) + D*tf.math.sin(2*np.pi*2/P*t) + E*1
        return y

# CREAMOS UNA INSTANCIA, ES DECIR UN MODELO
A = -0.1
B = 0.3
C = 0.05
D = 0.0
E = 14.75

my_linear_model = LinearModelV2(A,B,C,D,E)
#----------le pasamos ahora mil valores para graficarlo y observar como funciona-----------------------------------------
# ojo que aca graficamos directamente un tensor, pues matplotlib reconoce que por dentro tiene un valor de numpy 
period = 3.879
t = np.linspace(0, period, 1000)
y = my_linear_model(t)

plt.figure(figsize=(9, 6))
plt.plot(t, y)
plt.title('Salida del modelo con parámetros iniciales')
plt.gcf().patch.set_facecolor('white')
plt.show()

@tf.function 
def mse_error(prediction, target):
    return tf.reduce_mean((prediction - target)**2)

#Aumentando la tasa de convergencia obtuvimos una mejor aproximacion de los datos,
# pase de 0.01 a 0.05 y mejoró bastante
optimizer = tf.keras.optimizers.SGD(learning_rate=0.08, momentum=0.0)
#se optimiza usando el gradiente descendiente estocastico SGD, tasa de aprendizaje es un parametro importante

@tf.function
def train_step(xs, targets):
    with tf.GradientTape() as tape:
        predictions = my_linear_model(xs) #se hace una prediccion
        error = mse_error(predictions, targets) #se determina el error cuadratico
    gradients = tape.gradient(error, my_linear_model.parameters) #calculamos el gradiente
    print(gradients)
    optimizer.apply_gradients(zip(gradients, my_linear_model.parameters))


alerts = pd.read_pickle('alerts.pkl')
alerts.dropna(inplace=True)
slice_alerts = alerts[['mjd', 'magpsf_corr', 'sigmapsf_corr', 'fid']]
print(slice_alerts.head())
light_curve = slice_alerts.loc[['ZTF17aaajtgd']]
lc_g = light_curve[light_curve.fid == 1]
period = 3.879
print(lc_g)

#-------------------CONJUNTO DE ENTRENAMIENTO PARA QUE EL MODELO APRENDA-----------------------

t_train = tf.cast(lc_g.mjd%period,dtype=tf.float32)
train_y = lc_g.magpsf_corr 
train_y = train_y.astype(np.float32)
#----------------------------------------------------------------------------------------------
plt.figure(figsize=(9, 6))
plt.scatter(lc_g.mjd%period, train_y)
plt.title('training set for linear model')
plt.gcf().patch.set_facecolor('white')
plt.show()

#------------------------ENTRENAMOS EL CONJUNTO DE DATOS------------------------------------
iteration_log = []
error_log = []
for epoch in range(145):
    if epoch % 10 == 0:
        print('epoch', epoch)
    train_step(t_train, train_y)
    train_error = mse_error(my_linear_model(t_train), train_y)
    iteration_log.append(epoch)
    error_log.append(train_error)


fig = plt.figure(figsize=(10, 8))
fig.set_facecolor('white')
plt.scatter(lc_g.mjd%period, lc_g.magpsf_corr)


t_grid = np.linspace(0, period, 1000)
model_output = my_linear_model(t_grid) #graficamos el model output ya entrenado

plt.scatter(t_grid%period, model_output)
plt.show()

#-------------------CURVA DE APRENDIZAJE--------------------------------
plt.figure(figsize=(9, 6))
plt.plot(iteration_log, error_log)
plt.xlabel('Épocas o iteraciones')
plt.ylabel('MSE')
plt.title('Curva de aprendizaje @ conjunto de entrenamiento')
plt.gcf().patch.set_facecolor('white')
plt.show()


print("Parametros",my_linear_model.parameters)
'''
Parametros ListWrapper([<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.24151726>, 
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=0.24411476>, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-0.10106732>, 
<tf.Variable 'Variable:0' shape=() dtype=float32, numpy=-0.08176889>, <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=14.803932>])
Modelo entregado convergio a los siguientes valores
A=0.24151726
B=0.24411476
C=-0.10106732
D=-0.08176889
E=14.803932
'''