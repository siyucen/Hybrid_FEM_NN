import math
from typing import Iterator, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.sparse.linalg import splu
from skfem import *
from skfem.helpers import dot, grad
from skfem.models.poisson import laplace, mass
from skfem.visuals.matplotlib import plot

m = MeshTri().refined(8)
thermal_conductivity =lambda x,y:  2+math.sin(2*math.pi*x)*math.sin(2*math.pi*y)

heat_source=lambda x,y,t: (10+0*x*y)*t

ini_heat=lambda x,y: 4*(1-x)*x*(1-y)*y
e = ElementTriP1()
basis = Basis(m, e)

dt = 1e-4 #tau
Time=1
print('dt =', dt)


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

conductivity = basis.zeros() #conductivity.shape=(#DOF,)
source = basis.zeros() #conductivity.shape=(#DOF,)
ini=basis.zeros()

for i in range(0,np.size(conductivity)):
    conductivity[i] = thermal_conductivity(m.p[0,i],m.p[1,i])


source = basis.zeros()
source[:] = 10
f = asm(rhs, basis,source=basis.interpolate(source)) #source
f=np.reshape(f,(1,-1))
F=[]
for i in range(0,round(Time/dt)):
    t=i*dt
    F.append(t*f)

for j in range(0,np.size(conductivity)):
    ini[j] = ini_heat(m.p[0,j],m.p[1,j])



A = asm(conduction, basis, conductivity=basis.interpolate(conductivity)) #stiff


M = asm(mass, basis)    #mass

theta = 1    # BE
A0, M0 = penalize(A, M, D=basis.get_dofs())#Penalize degrees-of-freedom of a linear system by setting bdry node to lagre value
L = M0 + theta * A0 * dt #  left hand side,coeff of U^n
R = M0 - (1 - theta) * A0 * dt # right hand side, coeff of U^n-1

u_init = asm(ini_func, basis,ini_func=basis.interpolate(ini)) #source

backsolve = splu(L.T).solve  # .T as splu prefers CSC, compute LU-decomposition





def evolve(t: float, 
           u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:

    while round(t/dt)<len(F): #while t<1
        print(t/dt)
        f=F[round(t/dt)]
        f=np.reshape(f,(-1,))
        t, u = t + dt, backsolve(R*u+dt*f)
        yield t, u


U=[]
for i in evolve(0., u_init):
    U.append(i)
# U:list. Length=Nt.

print(U[round(Time/dt)-1])
x=U[round(Time/dt)-1][1]
plot(m, x, shading='gouraud', colorbar=True)
plt.show()




# construct observe data
U_new=[]
mesh_size=5
mesh = MeshTri().refined(mesh_size) #coarse mesh
xnew=np.linspace(0,1,np.power(2,mesh_size)+1)
ynew=np.linspace(0,1,np.power(2,mesh_size)+1)
T0=0.1 # observation interval= [0.9,1.0]
Nt=round(Time/dt)
Nt0=round((Time-T0)/dt)-1
for i in range(Nt0,Nt,50): # observation time step=dt*50
    U_dag=U[i][1]
    U_dag=U_dag[0:np.size(mesh.p,1)]
    f = interpolate.interp2d(mesh.p[0,:],mesh.p[1,:], U_dag, kind='linear')
    znew = f(xnew, ynew)
    znew=np.reshape(znew,(-1,))
    U_new.append(znew)





m = MeshTri.init_tensor(
    np.linspace(0, 1, np.power(2,mesh_size)+1) ,
    np.linspace(0, 1, np.power(2,mesh_size)+1 ))

   
plot(m, znew, shading='gouraud', colorbar=True)
plt.show()

# np.save("E:/project/fem_nn/fem_nn_2D_para/forward_data.npy",U_new)