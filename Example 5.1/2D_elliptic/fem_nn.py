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
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

#tf.keras.backend.set_floatx('float64')
tf.config.run_functions_eagerly(True)

def noise(u,delta):
    # r=np.random.rand(np.size(u)).astype(np.float64)
    r=np.random.normal(0,1,np.size(u))
    m=max(u)
    
    return u+delta*m*r

forward_data=np.load("E:/project/fem_nn/fem_nn_2D_elliptic/forward_data.npy")
forward_mesh=np.load("E:/project/fem_nn/fem_nn_2D_elliptic/forward_mesh.npy") #m.p on fine mesh
# forward_data=forward_data.reshape(forward_data.shape[0],1)
# qdag=lambda x,y: 1+y*(1-y)*np.sin(np.pi*x)
qdag =lambda x,y: 2+np.sin(2*np.pi*x)*np.sin(2*np.pi*y)


delta=1e-2
mesh_size=5#  h=1/2^mesh_size
reg_para=1e-8
reg_para_bdry=1e-6



mesh = MeshTri().refined(mesh_size)
e = ElementTriP1()
basis = Basis(mesh, e)
U_dag=forward_data[0:np.size(mesh.p,1)]# U on corase mesh ### Note here mesh index from corase to U_delta=noise(U_dag,delta) # shape(#DOF,)
U_delta=noise(U_dag,delta)

d=2
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

    
def error_U(U,Ud):
    return np.mean(np.square(U-Ud))

def q_H1(q,mesh):
    Nh=np.power(2,mesh_size)
    xnew = np.arange(0, 1, 1/Nh)
    ynew = np.arange(0, 1, 1/Nh)
    q=interpolate.interp2d(mesh.p[0,:],mesh.p[1,:],q,kind='linear')
    qnew = q(xnew, ynew)
    dqx=np.diff(qnew,axis=-1)
    dqx=np.append(dqx,np.reshape(2*dqx[:,np.size(dqx,1)-1]-dqx[:,np.size(dqx,1)-2],(-1,1)),axis=1)
    dqx=dqx*Nh
    dqy=np.diff(qnew,axis=0)
    dqy=np.append(dqy,np.reshape(2*dqy[np.size(dqy,0)-1,:]-dqy[np.size(dqy,0)-2,:],(1,-1)),axis=0)
    dqy=dqy*Nh

    return np.mean(np.square(dqx)+np.square(dqy))

def error_q(q,mesh):
    q_true=qdag(mesh.p[0,:],mesh.p[1,:]) # check dim
    dq=q-q_true
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
        # self.l2=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='sigmoid',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        # self.l3=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        self.l4=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        self.l5=keras.layers.Dense(1, dtype=tf.float64,activation=None,use_bias=True, kernel_initializer=keras.initializers.RandomUniform(0.001,0.125),bias_initializer=keras.initializers.RandomUniform(0.001,0.125))# type: ignore
            
       
        

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
       
       
        with tf.GradientTape(persistent=True) as tape:

            loss_equ=self(x)-2#-tf.sin(np.pi*x)
            

            loss=tf.reduce_mean(tf.square(loss_equ))
            
            
            
            grad_net=tape.gradient(loss,self.trainable_variables)
            self.optimizer.apply_gradients(zip(grad_net,self.trainable_variables))
            
            
            del tape

        return {"loss": loss}

    def save_model(self, path):
        self.net.save(path+"net")

    def call(self, x, training=None, mask=None):
        res=x
        # x=self.l1(x)
        # x=self.l2(x+res)
        # res1=x
        # x=self.l3(x)
        # x=self.l4(x+res1)
        # y=self.l5(x)
        x=self.l1(x+res)
        # x=self.l2(x)
        # x=self.l3(x)
        x=self.l4(x)
        y=self.l5(x)
        return y


class Net_fem(keras.Model):# type: ignore
    
    def __init__(self,
                 n_hidden_layers,
                 n_hidden_nodes
                 ):
        super(Net_fem, self).__init__() #initial step in class keras
        #your network
        self.l1=keras.layers.Dense(n_hidden_nodes,input_shape=(d,), dtype=tf.float64,activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        # self.l2=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='sigmoid',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        # self.l3=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        self.l4=keras.layers.Dense(n_hidden_nodes,dtype=tf.float64, activation='tanh',use_bias=True, kernel_initializer=keras.initializers.RandomNormal(),bias_initializer=keras.initializers.RandomNormal())
        self.l5=keras.layers.Dense(1, dtype=tf.float64,activation=None,use_bias=True, kernel_initializer=keras.initializers.RandomUniform(0.001,0.125),bias_initializer=keras.initializers.RandomUniform(0.001,0.125))# type: ignore
        
        
        

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
        x=data[0][0]   # shape(xxxx,2)


        
        y=np.transpose(mesh.p) #training_data # shape(xxxx,2)
        conductivity = self(y)
        conductivity_n = conductivity.numpy() 
        conductivity_n=np.reshape(conductivity_n,(-1,)) #conductivity.shape=(#DOF,)
        # rhs=data[1][0]
        with tf.GradientTape(persistent=True) as tape:

            # sample on boundary
            x1=x[:,0]
            x1=tf.reshape(x1,(-1,1))
            x2=tf.concat([x1,tf.ones(tf.shape(x1))],axis=1)
            x1=tf.concat([x1,tf.zeros(tf.shape(x1))],axis=1)
            y1=x[:,1]
            y1=tf.reshape(y1,(-1,1))
            y2=tf.concat([tf.ones(tf.shape(y1)),y1],axis=1)
            y1=tf.concat([tf.zeros(tf.shape(y1)),y1],axis=1)

            q1=self(x1)
            q2=self(x2)
            q3=self(y1)
            q4=self(y2)
            # conductivity = self(y)
            # conductivity_n = conductivity.numpy() 
            
            source_0 = basis.zeros() 


            A = asm(conduction, basis, conductivity=basis.interpolate(conductivity_n)) # stiffness matrix

            source_0[:]=10  # forward right hand side term 
            b = asm(rhs, basis,source=basis.interpolate(source_0))
            A, b = enforce(A, b, D=mesh.boundary_nodes())

            U= solve(A, b)#shape=(#DOF,)
            
            
            
            loss=error_U(U,U_delta)/2+reg_para/2*q_H1(conductivity_n,mesh)
            loss2=reg_para_bdry/2*(tf.reduce_mean(tf.square(q1-2))+tf.reduce_mean(tf.square(q2-2))+tf.reduce_mean(tf.square(q3-2))+tf.reduce_mean(tf.square(q4-2)))  
            

            DJ=grad_loss(U,A,conductivity_n) #shape=(#DOF,)
            DJ=np.reshape(DJ,(-1,1)) # shape=(#DOF,1)
            DJ_stop=tf.stop_gradient(DJ) #stop gradient
            grad_net=tape.gradient(tf.multiply(self(y),DJ_stop)+loss2,self.trainable_variables) 
            self.optimizer.apply_gradients(zip(grad_net,self.trainable_variables))

           
            # grad_net=tape.gradient(self(y)+loss2,self.trainable_variables) 
            # self.optimizer.apply_gradients(zip(grad_net,self.trainable_variables))
            
            
            
            del tape

        return {"loss": loss}

    def save_model(self, path):
        self.net.save(path+"net")

    def call(self, x, training=None, mask=None):
        res=x
        # x=self.l1(x)
        # x=self.l2(x+res)
        # res1=x
        # x=self.l3(x)
        # x=self.l4(x+res1)
        # y=self.l5(x)
        x=self.l1(x+res)
        # x=self.l2(x)
        # x=self.l3(x)
        x=self.l4(x)
        y=self.l5(x)
        return y



def grad_loss(U,A,q): 
    source_1 = basis.zeros() 
    source_1=U_delta-U
    b = asm(rhs, basis,source=basis.interpolate(source_1))
    A, b = enforce(A, b, D=mesh.boundary_nodes())
    V= solve(A, b) # adjoint solution
    Nh=np.power(2,mesh_size)
    xnew = np.arange(0, 1, 1/Nh)
    ynew = np.arange(0, 1, 1/Nh)
    grad_U_grad_V=FE_grad(U,V,mesh,xnew,ynew,Nh)
    lap_q=FE_laplace(q,mesh,xnew,ynew,Nh)
    return Riesz(grad_U_grad_V-reg_para*lap_q,xnew,ynew)

def FE_grad(U,V,mesh,xnew,ynew,Nh):
    U=interpolate.interp2d(mesh.p[0,:],mesh.p[1,:],U,kind='linear')
    V=interpolate.interp2d(mesh.p[0,:],mesh.p[1,:],V,kind='linear')
    unew = U(xnew, ynew)
    vnew = V(xnew, ynew)
    dux=np.diff(unew,axis=-1)
    dux=np.append(dux,np.reshape(2*dux[:,np.size(dux,1)-1]-dux[:,np.size(dux,1)-2],(-1,1)),axis=1)
    dux=dux*Nh
    duy=np.diff(unew,axis=0)
    duy=np.append(duy,np.reshape(2*duy[np.size(duy,0)-1,:]-duy[np.size(duy,0)-2,:],(1,-1)),axis=0)
    duy=duy*Nh
    dvx=np.diff(vnew,axis=-1)
    dvx=np.append(dvx,np.reshape(2*dvx[:,np.size(dvx,1)-1]-dvx[:,np.size(dvx,1)-2],(-1,1)),axis=1)
    dvx=dvx*Nh
    dvy=np.diff(vnew,axis=0)
    dvy=np.append(dvy,np.reshape(2*dvy[np.size(dvy,0)-1,:]-dvy[np.size(dvy,0)-2,:],(1,-1)),axis=0)
    dvy=dvy*Nh
    return dux*dvx+duy*dvy

def FE_laplace(q,mesh,xnew,ynew,Nh):
    q=interpolate.interp2d(mesh.p[0,:],mesh.p[1,:],q,kind='linear')
    qnew = q(xnew, ynew)
    dqx=np.diff(qnew,n=1,axis=-1)
    dqx=np.append(dqx,np.reshape(2*dqx[:,np.size(dqx,1)-1]-dqx[:,np.size(dqx,1)-2],(-1,1)),axis=1)
    dqx=dqx*Nh
    dqxx=np.diff(dqx,n=1,axis=-1)
    dqxx=np.append(dqxx,np.reshape(2*dqxx[:,np.size(dqxx,1)-1]-dqxx[:,np.size(dqxx,1)-2],(-1,1)),axis=1)
    dqxx=dqxx*Nh

    dqy=np.diff(qnew,n=1,axis=0)
    dqy=np.append(dqy,np.reshape(2*dqy[np.size(dqy,0)-1,:]-dqy[np.size(dqy,0)-2,:],(1,-1)),axis=0)
    dqy=dqy*Nh
    dqyy=np.diff(dqy,n=1,axis=0)
    dqyy=np.append(dqyy,np.reshape(2*dqyy[np.size(dqyy,0)-1,:]-dqyy[np.size(dqyy,0)-2,:],(1,-1)),axis=0)
    dqyy=dqyy*Nh
    return  dqxx+dqyy

def Riesz(g,x,y):
    f=interpolate.interp2d(x,y,g,kind='linear')
    
    # source_3=f(mesh.p[0,:],mesh.p[1,:])
    # source_3=np.diag(source_3)
    source_2 = basis.zeros() 

    
    for i in range(0,np.size(mesh.p,1)):
        source_2[i]=f(mesh.p[0,i],mesh.p[1,i])
    
    b = asm(rhs, basis,source=basis.interpolate(source_2))
    R, b = enforce(Rz, b, D=mesh.boundary_nodes())
    sol=solve(R,b)
    return sol






## prepare

test=Net_ini(2,16)
x_train=np.random.rand(200,d).astype(np.float64)
data=[x_train],[np.zeros_like(x_train[:,0])]
test.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3)) #compile model
test.fit(data[0], data[1], epochs=500,batch_size=40)
x_test=np.random.rand(200,d).astype(np.float64)
z=test(x_test)
print('initial net approximate constant=',tf.reduce_mean(tf.square(z-2)) )
### test_net train  function





## training data for fem_nn
x_train=mesh.p
x_train=np.transpose(x_train)
data=[x_train],[np.zeros_like(x_train[:,0])]

test_fem=Net_fem(2,16)
test_fem.l1=test.l1
test_fem.l4=test.l4
test_fem.l5=test.l5


# test with exact conductivity
# q_true=qdag(mesh.p[0,:],mesh.p[1,:])
# q=np.reshape(q_true,(-1,))
# source_0 = basis.zeros() 
# A = asm(conduction, basis, conductivity=basis.interpolate(q)) # stiffness matrix
# source_0[:]=10  # forward right hand side term 
# b = asm(rhs, basis,source=basis.interpolate(source_0))
# A, b = enforce(A, b, D=mesh.boundary_nodes())
# U= solve(A, b)
# print('L2_udag-udag=',error_U(U,U_dag)) ## initial error in L2-norm 
# print('loss with exact q: ',error_U(U,U_delta)/2+reg_para/2*q_H1(q,mesh))

## initial data
q=test_fem(np.transpose(mesh.p))
q=q.numpy() 
q=np.reshape(q,(-1,))
# q_true=qdag(mesh.p[0,:],mesh.p[1,:])
# q=np.reshape(q_true,(-1,))
plot(mesh, q, shading='gouraud', colorbar=True)
plt.title('Initial conductivity')
plt.show()
print('L2_q-qdag=',error_q(q,mesh))
source_0 = basis.zeros() 
A = asm(conduction, basis, conductivity=basis.interpolate(q)) # stiffness matrix
source_0[:]=10  # forward right hand side term 
b = asm(rhs, basis,source=basis.interpolate(source_0))
A, b = enforce(A, b, D=mesh.boundary_nodes())
U= solve(A, b)
print('L2_u-udag=',np.sqrt(error_U(U,U_dag))) ## initial error in L2-norm  



class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 2e-5):
            print("\n\n\nReached  loss value so cancelling training!\n\n\n")
            self.model.stop_training = True


test_fem.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2)) #compile model
test_fem.fit(data[0], data[1], epochs=20000,batch_size=n_sample,callbacks=haltCallback()) #train model




## terminal
q=test_fem(np.transpose(mesh.p)) 
q=q.numpy() 
q=np.reshape(q,(-1,)) 
q_true=qdag(mesh.p[0,:],mesh.p[1,:])
print('L2_q-qdag=',np.sqrt(error_q(q,mesh)))
plot(mesh, q_true, shading='gouraud', colorbar=True)
plt.title('Exact conductivity')
plt.show()
plot(mesh, q, shading='gouraud', colorbar=True)
plt.title('Recovered conductivity')
plt.show()


source_0 = basis.zeros()
A = asm(conduction, basis, conductivity=basis.interpolate(q)) # stiffness matrix
source_0[:]=10  # forward right hand side term 
b = asm(rhs, basis,source=basis.interpolate(source_0))
A, b = enforce(A, b, D=mesh.boundary_nodes())
U= solve(A, b)
print('L2_u-udag=',np.sqrt(error_U(U,U_dag)))
plot(mesh, U_dag, shading='gouraud', colorbar=True)
plt.title('Exact solution')
plt.show()
plot(mesh, U, shading='gouraud', colorbar=True)
plt.title('Recovered solution')
plt.show()


