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
from typing import Iterator, Tuple
from scipy.sparse.linalg import splu
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import datetime
from skfem import *
from skfem.helpers import dot, grad
from skfem.models.poisson import  mass
from skfem.visuals.matplotlib import plot
from matplotlib.pyplot import subplots
import os
os.environ["KMP_DUPLICATE_LIB_OK"]  =  "TRUE"

plt.rcParams.update({'font.size': 16})

#tf.keras.backend.set_floatx('float64')
tf.config.run_functions_eagerly(True)

def noise(U,delta):
    N=np.size(U,0)
    for i in range(0,N):
        u=U[i,:]
        r=np.random.normal(0,1,np.size(u))
        m=max(u)
        U[i,:]=u+delta*m*r
    return U

forward_data=np.load("E:/project/fem_nn/fem_nn_2D_para/forward_data.npy") #array (time,spcae) on tensor mesh


qdag =lambda x,y: 2+np.sin(2*np.pi*x)*np.sin(2*np.pi*y)

heat_source=lambda x,y,t: (10+0*x*y)*t
ini_heat=lambda x,y: 4*(1-x)*x*(1-y)*y

delta=5e-2
mesh_size=5#  h=1/2^mesh_size
c_0=0.5
c_1=5 # cut-off


Time=1
T0=0.1
dt=1/200
reg_para=1e-7
reg_para_bdry=1e-6
Nh=np.power(2,mesh_size)+1


mesh = MeshTri.init_tensor(
    np.linspace(0, 1, Nh) ,
    np.linspace(0, 1, Nh ))
e = ElementTriP1()
basis = Basis(mesh, e)
U_dag=forward_data
U_delta=noise(U_dag.copy(),delta)


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

@LinearForm
def ini_func(v, w):
    return w['ini_func'] * v




Rz=asm(laplace,basis)


source = basis.zeros() #conductivity.shape=(#DOF,)
ini=basis.zeros()
u_init0=asm(ini_func, basis,ini_func=basis.interpolate(ini)) #zero initial value
source[:] = 10
f = asm(rhs, basis,source=basis.interpolate(source)) #source
f=np.reshape(f,(1,-1))
F=[]
for i in range(0,round(Time/dt)):
    t=i*dt
    F.append(t*f)

for j in range(0,np.size(mesh.p,1)):
    ini[j] = ini_heat(mesh.p[0,j],mesh.p[1,j])
u_init = asm(ini_func, basis,ini_func=basis.interpolate(ini)) #initial value

M = asm(mass, basis) 



def error_U(U,Ud):

    return np.mean(np.square(U-Ud))

def q_H1(q):
    qnew=np.reshape(q,(Nh,Nh))
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

def forward_solver(q): # get forward data (size=U_delta) from guess q
    A = asm(conduction, basis, conductivity=basis.interpolate(q)) # stiffness matrix
    theta = 1    # BE
    A0, M0 = penalize(A, M, D=basis.get_dofs())#Penalize degrees-of-freedom of a linear system by setting bdry node to lagre value
    L = M0 + theta * A0 * dt #  left hand side,coeff of U^n
    R = M0 - (1 - theta) * A0 * dt # right hand side, coeff of U^n-1

    backsolve = splu(L.T).solve
            
    def evolve(t: float, 
               u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:
        while round(t/dt)<len(F): #while t<1
            f=F[round(t/dt)]
            f=np.reshape(f,(-1,))
            t, u = t + dt, backsolve(R*u+dt*f)
            yield t, u
    U_1=[]
    for i in evolve(0., u_init):
        U_1.append(i)
    U=np.reshape(U_1[round((Time-T0)/dt)-1][1],(1,-1))
    for i in range(round((Time-T0)/dt),round((Time)/dt)):
        U=np.append(U,np.reshape(U_1[i][1],(1,-1)),axis=0)
    return U,A


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
            
            


            # A = asm(conduction, basis, conductivity=basis.interpolate(conductivity_n)) # stiffness matrix


            # theta = 1    # BE
            # A0, M0 = penalize(A, M, D=basis.get_dofs())#Penalize degrees-of-freedom of a linear system by setting bdry node to lagre value
            # L = M0 + theta * A0 * dt #  left hand side,coeff of U^n
            # R = M0 - (1 - theta) * A0 * dt # right hand side, coeff of U^n-1

            

            # backsolve = splu(L.T).solve
            
            # def evolve(t: float, 
            #            u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:

            #     while round(t/dt)<len(F): #while t<1
            #         print(t/dt)
            #         f=F[round(t/dt)]
            #         f=np.reshape(f,(-1,))
            #         t, u = t + dt, backsolve(R*u+dt*f)
            #         yield t, u
            # U_1=[]
            # for i in evolve(0., u_init):
            #     U_1.append(i)
            # # U_1 list, observe (0,Time)
            # U=np.reshape(U_1[round((Time-T0)/dt)-1][1],(1,-1))
            # for i in range(round((Time-T0)/dt),round((Time)/dt)):
            #     U=np.append(U,np.reshape(U_1[i][1],(1,-1)),axis=0)
            # # U array, size=U_delta
            c=np.ones((np.shape(conductivity_n)))
            c0=c*c_0
            c1=c*c_1
            conductivity_n=np.maximum(np.minimum(conductivity_n,c1),c0)
            U,A=forward_solver(conductivity_n)
            loss=error_U(U,U_delta)/2+reg_para/2*q_H1(conductivity_n)

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
    ## backward solve adjoint equation
    source_1=U_delta-U # shape(#t,#mesh) from T-T0 to T
    F1=[]
    for i in range(0,round(T0/dt)+1):
        t=i*dt
        source=source_1[round(T0/dt)-i]
        f = asm(rhs, basis,source=basis.interpolate(source)) #source
        f=np.reshape(f,(1,-1))
        F1.append(f)

    theta = 1    # BE
    A0, M0 = penalize(A, M, D=basis.get_dofs())
    L = M0 + theta * A0 * dt 
    R = M0 - (1 - theta) * A0 * dt 
    backsolve = splu(L.T).solve
    
    def evolve(t: float, 
               u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:

        while round(t/dt)<len(F1): #while t<1
  
            f=F1[round(t/dt)]
            f=np.reshape(f,(-1,))
            t, u = t + dt, backsolve(R*u+dt*f)
            yield t, u

    V_1=[]
    for i in evolve(0., u_init0):
        V_1.append(i)
    V=np.reshape(V_1[round((T0)/dt)][1],(1,-1))
    for i in range(0,round((T0)/dt)):
        V=np.append(V,np.reshape(V_1[round((T0)/dt)-1-i][1],(1,-1)),axis=0)
    

    grad_U_grad_V=FE_grad(U,V,Nh) #(time,mesh)
    lap_q=FE_laplace(q,Nh)
    return Riesz(grad_U_grad_V-reg_para*lap_q)

def FE_grad(U,V,Nh):
    unew = np.reshape(U,(-1,Nh,Nh))
    vnew = np.reshape(V,(-1,Nh,Nh))
    dux=np.diff(unew,axis=-1)
    dux=np.append(dux,np.reshape(2*dux[:,:,np.size(dux,2)-1]-dux[:,:,np.size(dux,2)-2],(-1,Nh,1)),axis=2)
    dux=dux*Nh
    duy=np.diff(unew,axis=-2)
    duy=np.append(duy,np.reshape(2*duy[:,np.size(duy,1)-1,:]-duy[:,np.size(duy,1)-2,:],(-1,1,Nh)),axis=1)
    duy=duy*Nh

    dvx=np.diff(vnew,axis=-1)
    dvx=np.append(dvx,np.reshape(2*dvx[:,:,np.size(dvx,2)-1]-dvx[:,:,np.size(dvx,2)-2],(-1,Nh,1)),axis=2)
    dvx=dvx*Nh
    dvy=np.diff(vnew,axis=-2)
    dvy=np.append(dvy,np.reshape(2*dvy[:,np.size(dvy,1)-1,:]-dvy[:,np.size(dvy,1)-2,:],(-1,1,Nh)),axis=1)
    dvy=dvy*Nh
    return np.mean(dux*dvx+duy*dvy,axis=0)

def FE_laplace(q,Nh):
    qnew = np.reshape(q,(Nh,Nh))
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

def Riesz(g):

    f=np.reshape(g,(-1,1))
    source_2 = basis.zeros() 
    for i in range(0,np.size(mesh.p,1)):
        source_2[i]=f[i]
    
    b = asm(rhs, basis,source=basis.interpolate(source_2))
    R, b = enforce(Rz, b, D=mesh.boundary_nodes())
    sol=solve(R,b)
    return sol






## prepare

test=Net_ini(2,32)
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

test_fem=Net_fem(2,32)
test_fem.l1=test.l1
test_fem.l4=test.l4
test_fem.l5=test.l5


## initial data
q=test_fem(np.transpose(mesh.p))
q=q.numpy() 
q=np.reshape(q,(-1,))
# q_true=qdag(mesh.p[0,:],mesh.p[1,:])
# q=np.reshape(q_true,(-1,))
plot(mesh, q, shading='gouraud', colorbar=True)
plt.title('Initial conductivity')
plt.show()



print('L2_q-qdag=',np.sqrt(error_q(q,mesh)))


U,A=forward_solver(q)
print('L2_u-udag=',np.sqrt(error_U(U,U_dag))) ## initial error in L2-norm  
 


class haltCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('loss') <= 7e-8):
            print("\n\n\nReached  loss value so cancelling training!\n\n\n")
            self.model.stop_training = True


test_fem.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-2)) #compile model
test_fem.fit(data[0], data[1], epochs=25000,batch_size=n_sample,callbacks=haltCallback()) #train model
for Iter in range(0,20):
    test_fem.fit(data[0], data[1], epochs=250,batch_size=n_sample,callbacks=haltCallback()) #train model
    q=test_fem(np.transpose(mesh.p)) 
    q=q.numpy() 
    q=np.reshape(q,(-1,)) 
    q_true=qdag(mesh.p[0,:],mesh.p[1,:])
    print('L2_q-qdag=',np.sqrt(error_q(q,mesh)))    
    fig, ax = subplots()
    ax.set_ylim((0.,1.))
    ax = plot(mesh, q_true,ax=ax,vmin=1,vmax=3, shading='gouraud', colorbar=True)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    fig, ax = subplots()
    ax.set_ylim((0.,1.))
    plot(mesh, q ,ax=ax, shading='gouraud', colorbar=True)
    plt.title('Recovered conductivity')
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()
    fig, ax = subplots()
    ax.set_ylim((0.,1.))
    plot(mesh, q,ax=ax,vmin=1,vmax=3.0001, shading='gouraud', colorbar=True)
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.show()


## terminal
q=test_fem(np.transpose(mesh.p)) 
q=q.numpy() 
q=np.reshape(q,(-1,)) 
q_true=qdag(mesh.p[0,:],mesh.p[1,:])
print('L2_q-qdag=',np.sqrt(error_q(q,mesh)))
Ntest=1000
x2=np.linspace(0,1,Ntest)
y2=np.linspace(0,1,Ntest)
[x2,y2]=np.meshgrid(x2,y2)
qd2=qdag(x2,y2)
qd2=np.reshape(qd2,(-1,))
mesh2 = MeshTri.init_tensor(
    np.linspace(0, 1, Ntest) ,
    np.linspace(0, 1, Ntest ))
q2=test_fem(np.transpose(mesh2.p)) 
q2=q2.numpy() 
q2=np.reshape(q2,(-1,))
print('L2_q-qdag_2=',np.sqrt(np.mean(np.square(q2-qd2)))) 


fig, ax = subplots()
ax.set_ylim((0.,1.))
ax = plot(mesh, q_true,ax=ax,vmin=1,vmax=3, shading='gouraud', colorbar=True)
plt.title('Exact conductivity')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
fig, ax = subplots()
ax.set_ylim((0.,1.))
plot(mesh, q ,ax=ax, shading='gouraud', colorbar=True)
plt.title('Recovered conductivity')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
fig, ax = subplots()
ax.set_ylim((0.,1.))
plot(mesh, q,ax=ax,vmin=1,vmax=3.30001, shading='gouraud', colorbar=True)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()






U,A=forward_solver(q)
print('L2_u-udag=',np.sqrt(error_U(U,U_dag)))
U=U[np.size(U,0)-1,:]
U_dag=U_dag[np.size(U_dag,0)-1,:]
fig, ax = subplots()
ax.set_ylim((0.,1.))
plot(mesh, U_dag,ax=ax, shading='gouraud', colorbar=True)
plt.title('Exact solution')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()
fig, ax = subplots()
ax.set_ylim((0.,1.))
plot(mesh, U,ax=ax, shading='gouraud', colorbar=True)
plt.title('Recovered solution')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()


