import math
import numpy as np
from skfem import *
from skfem.helpers import dot, grad
from skfem.visuals.matplotlib import plot
from scipy import interpolate
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots

# create the mesh
m = MeshTri().refined(8)

# thermal_conductivity =lambda x,y: 1+y*(1-y)*math.sin(math.pi*x)
thermal_conductivity =lambda x,y: 2+math.sin(2*math.pi*x)*math.sin(2*math.pi*y)

e = ElementTriP1()
basis = Basis(m, e)

# this method could also be imported from skfem.models.laplace
@BilinearForm
def conduction(u, v, w):
    return dot(w['conductivity'] * grad(u), grad(v))


# this method could also be imported from skfem.models.unit_load
@LinearForm
def rhs(v, w):
    return w['source'] * v



conductivity = basis.zeros() #conductivity.shape=(#DOF,)
source = basis.zeros() #conductivity.shape=(#DOF,)
for i in range(0,np.size(conductivity)):
    conductivity[i] = thermal_conductivity(m.p[0,i],m.p[1,i])
source[:] = 10

A = asm(conduction, basis, conductivity=basis.interpolate(conductivity))


b = asm(rhs, basis,source=basis.interpolate(source))



# enforce Dirichlet boundary conditions
A, b = enforce(A, b, D=m.boundary_nodes()) # shape not change

# solve -- can be anything that takes a sparse matrix and a right-hand side
x = solve(A, b)





plot(m, x, shading='gouraud', colorbar=True)

plt.show()

mesh_size=5
mesh = MeshTri().refined(mesh_size) #coarse mesh
U_dag=x[0:np.size(mesh.p,1)]   # forward data

f = interpolate.interp2d(mesh.p[0,:],mesh.p[1,:], U_dag, kind='linear')
xnew=np.linspace(0,1,np.power(2,mesh_size)+1)
ynew=np.linspace(0,1,np.power(2,mesh_size)+1)
znew = f(xnew, ynew)
znew=np.reshape(znew,(-1,))
m = MeshTri.init_tensor(
    np.linspace(0, 1, np.power(2,mesh_size)+1) ,
    np.linspace(0, 1, np.power(2,mesh_size)+1 ))

   
plot(m, znew, shading='gouraud', colorbar=True)
plt.show()

fig, ax = subplots()

ax.set_ylim((0.,1.))
plot(m, znew, ax=ax,shading='gouraud', colorbar=True)
plt.xlim(0,1)
plt.ylim(0,1)
plt.show()

# np.save("E:/project/fem_nn/fem_nn_2D_elliptic/forward_data.npy",znew)
