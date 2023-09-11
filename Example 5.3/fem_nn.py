import imp
from traceback import print_tb
from turtle import forward
import numpy as np
import pandas as pd
import math
import tensorflow as tf
import scipy as sci
from tensorflow import keras
from scipy import special as sp
from scipy import interpolate
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
from skfem import *
from skfem.helpers import dot, grad
from skfem.visuals.matplotlib import plot
from matplotlib.pyplot import subplots
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

tf.keras.backend.set_floatx('float64')
tf.config.run_functions_eagerly(True)

def noise(u,delta):
    # r=np.random.rand(np.size(u)).astype(np.float64)
    r=np.random.normal(0,1,np.size(u))
    r=np.reshape(r,(1,-1))
    m=max(u)
    
    return u+delta*m*r

forward_data=np.load("E:/project/fem_nn/fem_nn_partial_data_1D_elliptic/forward_data.npy")
forward_mesh=np.load("E:/project/fem_nn/fem_nn_partial_data_1D_elliptic/forward_mesh.npy") #m.p on fine mesh

qdag=lambda x: 2+10*(1-x)*np.power(x,2)

delta=1e-3

reg_para=1e-6
reg_para_bdry=1e-4

mesh_size=1/40

c_0=0.5
c_1=5 # cut-off

observe_left=0.3
observe_right=0.7
# observe_left=0
# observe_right=1

U_delta=noise(forward_data,delta)
U_delta=np.reshape(U_delta,(-1,)) # fine mesh (1000,)


mesh = MeshLine(np.linspace(0, 1,round(1/mesh_size)))  # mesh size
e = ElementLineP1()
basis = Basis(mesh, e)
forward_func=interpolate.interp1d(np.reshape(forward_mesh,(-1,)),forward_data)
U_dag=forward_func(mesh.p)# U on corase mesh ### Note here mesh index from corase to U_delta=noise(U_dag,delta) # shape(#DOF,)

U_delta_2=interpolate.interp1d(np.reshape(forward_mesh,(-1,)),U_delta)
U_delta_2=U_delta_2(mesh.p) #(1,40)

d=1
n_sample=np.size(mesh.p,1) # sampling on FEM nodes




@BilinearForm
def conduction(u, v, w):
    return dot(w['conductivity'] * grad(u), grad(v))

@BilinearForm
def laplace(u, v, _):
    return dot( grad(u), grad(v))+dot(u,v)

# this method could also be imported from skfem.models.unit_load
@LinearForm
def rhs(v, w):
    return w['source'] * v




Rz=asm(laplace,basis)


def func_cut_off(f):
    # get the cut-off of a function f on the subdomain
    # f is given on mesh point
    g=f.copy()
    for i in range(0,f.size):
        x=mesh.p[0,i]
        if ((x<observe_left)|(x>observe_right)):
            g[i]=0
    return g

    
def error_U(U,Ud):
    return np.mean(np.square(U-Ud))

def q_H1(q,mesh):
    

    dqx=np.diff(q,axis=-1)
    dqx=np.append(dqx,np.reshape(2*dqx[np.size(dqx,0)-1]-dqx[np.size(dqx,0)-2],(-1,)),axis=0)
    Nh=round(1/mesh_size)
    dqx=dqx*Nh
    return np.mean(np.square(dqx))

def error_q(q,mesh):
    q_true=qdag(mesh.p[0,:]) # check dim
    dq=q-q_true
    # try element quadrature
    return np.mean(np.square(dq))
def error_dU(q):
    N=q.shape[0]
    h=1/N
    dx0=tf.reshape((q[1,0]-q[0,0])/h,(1,1))
    dxN=tf.reshape((q[N-1,0]-q[N-2,0])/h,(1,1))
    
    q1=q[0:N-3,0]
    q2=q[2:N-1,0]
    dx=tf.reshape((q1-q2)/(2*h),(-1,1))
    dq=tf.concat([dx0,dx,dxN],axis=0)
    a=tf.reduce_mean(tf.square(dq))
    b=tf.reduce_mean(tf.square(q))
    return a+b   #tf.square(tf.linalg.norm(dU,2))*h 

def error_qd(q1,q2):
    
    dq=q1-q2
    # try element quadrature
    return np.mean(np.square(dq))



class Net_ini(keras.Model):# type: ignore
    
    def __init__(self,
                 n_hidden_layers,
                 n_hidden_nodes
                 ):
        super(Net_ini, self).__init__() #initial step in class keras
        #your network
        self.l1=keras.layers.Dense(n_hidden_nodes,input_shape=(d,), dtype=tf.float64,activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        # self.l2=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        # self.l3=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        # self.l4=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        self.l5=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        self.l6=keras.layers.Dense(1, dtype=tf.float64,activation=None,use_bias=True, kernel_initializer=keras.initializers.RandomUniform(0.001,0.125),bias_initializer=keras.initializers.RandomUniform(0.001,0.125))# type: ignore
            
       
        

    def compile(
            self,
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.mean_squared_error,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            jit_compile=None,
            **kwargs,
    ):
        super(Net_ini, self).compile() #initial step in class keras
        self.optimizer = optimizer
        self.loss_fn = loss


    def train_step(self, data):
        x=data[0][0]
        global loss_value
        # lo=tf.zeros((1,1),dtype=tf.float64)
        with tf.GradientTape(persistent=True) as tape:

            loss_equ=self(x)-2#-tf.sin(2*np.pi*x)
            

            loss=tf.reduce_mean(tf.square(loss_equ))
            
            
            grad_net=tape.gradient(loss,self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad_net,self.trainable_variables))
            
            
            del tape

        return {"loss": loss}


    def save_model(self, path):
        self.net.save(path+"net")

    def call(self, x, training=None, mask=None):
        x=self.l1(x)
        # x=self.l2(x)
        # x=self.l3(x)
        # x=self.l4(x)
        x=self.l5(x)
        y=self.l6(x)
        return y

loss_value=tf.zeros((1,1),dtype=tf.float64)
error_value=tf.zeros((1,1),dtype=tf.float64)
class Net_fem(keras.Model):# type: ignore
    
    def __init__(self,
                 n_hidden_layers,
                 n_hidden_nodes
                 ):
        super(Net_fem, self).__init__() #initial step in class keras
        #your network
        self.l1=keras.layers.Dense(n_hidden_nodes,input_shape=(d,), dtype=tf.float64,activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        # self.l2=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        # self.l3=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        # self.l4=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        self.l5=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        self.l6=keras.layers.Dense(1, dtype=tf.float64,activation=None,use_bias=True, kernel_initializer=keras.initializers.RandomUniform(0.001,0.125),bias_initializer=keras.initializers.RandomUniform(0.001,0.125))# type: ignore
        
        
        

    def compile(
            self,
            optimizer=keras.optimizers.Adam(),
            loss=keras.losses.mean_squared_error,
            metrics=None,
            loss_weights=None,
            weighted_metrics=None,
            run_eagerly=None,
            steps_per_execution=None,
            jit_compile=None,
            **kwargs,
    ):
        super(Net_fem, self).compile() #initial step in class keras
        self.optimizer = optimizer
        self.loss_fn = loss


    def train_step(self, data):
        global loss_value, error_value,qd
        x=data[0][0]   # shape(xxxx,2)


        
        y=np.transpose(mesh.p) #training_data # shape(xxxx,2)
        conductivity = self(y)
        conductivity_n = conductivity.numpy() 
        conductivity_n=np.reshape(conductivity_n,(-1,)) #conductivity.shape=(#DOF,)
         
        err=np.sqrt(error_qd(conductivity_n,qd))
        error_value=np.append(error_value,[err])
        # rhs=data[1][0]
        with tf.GradientTape(persistent=True) as tape:

            # sample on boundary

            q1=self(tf.zeros((1,1)))
            q2=self(tf.ones((1,1)))

            # conductivity = self(y)
            # conductivity_n = conductivity.numpy() 
            
            source_0 = basis.zeros() 

            c=np.ones((np.shape(conductivity_n)))
            c0=c*c_0
            c1=c*c_1
            conductivity_n=np.maximum(np.minimum(conductivity_n,c1),c0)
            A = asm(conduction, basis, conductivity=basis.interpolate(conductivity_n)) # stiffness matrix

            source_0[:]=10  # forward right hand side term 
            b = asm(rhs, basis,source=basis.interpolate(source_0))
            A, b = enforce(A, b, D=mesh.boundary_nodes())

            U= solve(A, b)#shape=(#DOF,)
            
            

            loss=error_U(func_cut_off(U),func_cut_off(np.reshape(U_delta_2,(-1,))))/2+reg_para/2*q_H1(conductivity_n,mesh)
            loss2=reg_para_bdry/2*(tf.reduce_mean(tf.square(q1-2))+tf.reduce_mean(tf.square(q2-2)) )
            loss_reshaped=tf.reshape(loss+loss2,(1,1))
            loss_value=tf.concat([loss_value,loss_reshaped],axis=0)

            DJ=grad_loss(U,A,conductivity_n) #shape=(#DOF,)
            DJ=np.reshape(DJ,(-1,1)) # shape=(#DOF,1)
            DJ_stop=tf.stop_gradient(DJ) #stop gradient
            grad_net=tape.gradient(tf.multiply(self(y),DJ_stop)+loss2,self.trainable_variables) 
            self.optimizer.apply_gradients(zip(grad_net,self.trainable_variables))
            
            
            
            del tape

        return {"loss": loss}

    def save_model(self, path):
        self.net.save(path+"net")

    def call(self, x, training=None, mask=None):
        # res=x
        x=self.l1(x)
        # x=self.l2(x)
        # x=self.l3(x)
        # x=self.l4(x)
        x=self.l5(x)
        y=self.l6(x)
        return y



def grad_loss(U,A,q): 
    Nh=round(1/mesh_size)
    
    U_func=interpolate.interp1d(np.reshape(mesh.p,(-1,)),U)
    Uf=U_func(forward_mesh)
    source_1=U_delta-Uf
    s=np.reshape(source_1,(Nh,25))
    ss=np.mean(s,axis=1)
    ss=func_cut_off(ss)
    b = asm(rhs, basis,source=basis.interpolate(ss))
    A, b = enforce(A, b, D=mesh.boundary_nodes())
    V= solve(A, b) # adjoint solution
    
   

    grad_U_grad_V=FE_grad(U,V,Nh)
    lap_q=FE_laplace(q,Nh)

    return Riesz(grad_U_grad_V-reg_para*lap_q)
    # return Riesz(grad_U_grad_V)

def FE_grad(U,V,Nh):
    # U=interpolate.interp1d(np.reshape(mesh.p,(-1,)),U,kind='linear')
    # V=interpolate.interp1d(np.reshape(mesh.p,(-1,)),V,kind='linear')
    unew = U
    vnew = V
    dux=np.diff(unew,axis=-1)

    dux=np.append(dux,np.reshape(2*dux[np.size(dux,0)-1]-dux[np.size(dux,0)-2],(-1,)),axis=0)
    dux=dux*Nh
   
    dvx=np.diff(vnew,axis=-1)
    dvx=np.append(dvx,np.reshape(2*dvx[np.size(dvx,0)-1]-dvx[np.size(dvx,0)-2],(-1,)),axis=0)
    dvx=dvx*Nh
    
    return dux*dvx

def FE_laplace(q,Nh):
    # q=interpolate.interp1d(np.reshape(mesh.p,(-1,)),q,kind='linear')
    qnew = q
    dqx=np.diff(qnew,n=1,axis=-1)
    dqx=np.append(dqx,np.reshape(2*dqx[np.size(dqx,0)-1]-dqx[np.size(dqx,0)-2],(-1,)),axis=0)
    dqx=dqx*Nh
    dqxx=np.diff(dqx,n=1,axis=-1)
    dqxx=np.append(dqxx,np.reshape(2*dqxx[np.size(dqxx,0)-1]-dqxx[np.size(dqxx,0)-2],(-1,)),axis=0)
    dqxx=dqxx*Nh

    
    return  dqxx

def Riesz(g):
    # f=interpolate.interp1d(x,g,kind='linear')
    f=np.reshape(g,(-1,1))
    source_2 = basis.zeros() 

    
    for i in range(0,np.size(mesh.p,1)):
        source_2[i]=f[i]
    
    b = asm(rhs, basis,source=basis.interpolate(source_2))
    R, b = enforce(Rz, b, D=mesh.boundary_nodes())
    sol=solve(R,b)
    return sol



plt.rcParams.update({'font.size': 16})


## prepare

test=Net_ini(2,32)
x_train=np.random.rand(200,d).astype(np.float64)
data=[x_train],[np.zeros_like(x_train[:,0])]
test.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3)) #compile model
test.fit(data[0], data[1], epochs=500,batch_size=40)




x_test=np.random.rand(200,d).astype(np.float64)
x_test=np.sort(x_test,axis=0)
z=test(x_test)
plt.plot(x_test,z)
plt.show()
print('initial net approximate constant=',tf.reduce_mean(tf.square(z-2)) )






## training data for fem_nn
x_train=mesh.p
x_train=np.transpose(x_train)
data=[x_train],[np.zeros_like(x_train[:,0])]

test_fem=Net_fem(2,32)
test_fem.l1=test.l1
# test_fem.l2=test.l2
# test_fem.l3=test.l3
# test_fem.l4=test.l4
test_fem.l5=test.l5
test_fem.l6=test.l6



## initial data
x_test=np.linspace(0, 1,round(1/mesh_size))
qd=qdag(x_test)
q=test_fem(np.transpose(mesh.p))
q=q.numpy() 
q=np.reshape(q,(-1,))
plt.rcParams.update({'font.size': 16})
plt.plot(x_test,q)
plt.plot(x_test,qd)
plt.title('Initial conductivity')
plt.show()
print('L2_q-qdag=',np.sqrt(error_qd(q,qd)))
source_0 = basis.zeros() 
A = asm(conduction, basis, conductivity=basis.interpolate(q)) # stiffness matrix
source_0[:]=10  # forward right hand side term 
b = asm(rhs, basis,source=basis.interpolate(source_0))
A, b = enforce(A, b, D=mesh.boundary_nodes())
U= solve(A, b)
print('L2_u-udag=',np.sqrt(error_U(U,U_dag))) ## initial error in L2-norm  



# class haltCallback(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs={}):
#         if(logs.get('loss') <= 2e-8):
#             print("\n\n\nReached loss value so cancelling training!\n\n\n")
#             self.model.stop_training = True


test_fem.compile(optimizer=keras.optimizers.Adam(beta_2=0.9,learning_rate=1e-3)) #compile model
# test_fem.fit(data[0], data[1], epochs=200,batch_size=n_sample,callbacks=haltCallback())
# loss_value2=loss_value.numpy()
# loss_value2=loss_value2[1:np.size(loss_value2,0)]
# fig, ax = subplots()
# ax.set_yscale('log')
# ax.plot(loss_value2)
# plt.show()
# test_fem.fit(data[0], data[1], epochs=10000,batch_size=n_sample,callbacks=haltCallback()) #train model
test_fem.fit(data[0], data[1], epochs=2000,batch_size=n_sample) #train model
# for Iter in range(0,19):
#     test_fem.fit(data[0], data[1], epochs=200,batch_size=n_sample,callbacks=haltCallback()) #train model
#     q=test_fem(np.transpose(mesh.p)) 
#     q=q.numpy() 
#     q=np.reshape(q,(-1,)) 
#     print('L2_q-qdag=',np.sqrt(error_qd(q,qd)))

#     loss_value2=loss_value.numpy()
#     loss_value2=loss_value2[1:np.size(loss_value2,0)]
#     fig, ax = subplots()
#     ax.set_yscale('log')
#     ax.plot(loss_value2)
#     plt.show()

#     plt.plot(x_test,q,label='recovered')
#     plt.plot(x_test,qd,label='exact')
#     plt.legend()
#     plt.xlim(0,1)
#     plt.ylim(0.5,3.5)
#     plt.show()
# test_fem.fit(data[0], data[1], epochs=8000,batch_size=n_sample,callbacks=haltCallback()) #train model

loss_value2=loss_value.numpy()
loss_value2=loss_value2[1:np.size(loss_value2,0)]
fig, ax = subplots()
ax.set_yscale('log')
# ax.set_xscale('log')
ax.plot(loss_value2)
plt.show()
np.save("E:/project/fem_nn/fem_nn_partial_data_1D_elliptic/loss_1D_2_32_1e-2.npy",loss_value2)
error_value=error_value[1:np.size(error_value,0)]
fig, ax = subplots()
ax.plot(error_value)
plt.show()
np.save("E:/project/fem_nn/fem_nn_partial_data_1D_elliptic/error_1D_2_32_1e-2.npy",error_value)

## terminal
q=test_fem(np.transpose(mesh.p)) 
q=q.numpy() 
q=np.reshape(q,(-1,)) 
np.savetxt("E:/project/fem_nn/fem_nn_partial_data_1D_elliptic/q.txt", q)

print('L2_q-qdag=',np.sqrt(error_qd(q,qd)))

plt.plot(x_test,q,label='recovered')
plt.plot(x_test,qd,label='exact')
plt.legend()
plt.xlim(0,1)
plt.ylim(0.5,3.5)
plt.show()




source_0 = basis.zeros()
A = asm(conduction, basis, conductivity=basis.interpolate(q)) # stiffness matrix
source_0[:]=10  # forward right hand side term 
b = asm(rhs, basis,source=basis.interpolate(source_0))
A, b = enforce(A, b, D=mesh.boundary_nodes())
U= solve(A, b)
print('L2_u-udag=',np.sqrt(error_U(U,U_dag)))
plt.plot(x_test,U)
plt.plot(x_test,np.reshape(U_dag,(-1,)))
plt.title('Recovered solution')
plt.show()



