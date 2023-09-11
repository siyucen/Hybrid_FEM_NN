import math
import numpy as np
from skfem import *
from skfem.helpers import dot, grad


# create the mesh
x = np.linspace(0, 1,1000)
m = MeshLine(x)
# m1 = MeshLine(np.linspace(0, 5, 50))
# m2 = MeshLine(np.linspace(0, 1, 10))
# m = (m1 * m2).with_boundaries(
#     {
#         "left": lambda x: x[0] == 0.0
#     }
# )
thermal_conductivity =lambda x: 2+10*(1-x)*np.power(x,2)
e = ElementLineP1()
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
    conductivity[i] = thermal_conductivity(m.p[0,i])
source[:] = 10

A = asm(conduction, basis, conductivity=basis.interpolate(conductivity))


b = asm(rhs, basis,source=basis.interpolate(source))



# enforce Dirichlet boundary conditions
A, b = enforce(A, b, D=m.boundary_nodes()) # shape not change

# solve -- can be anything that takes a sparse matrix and a right-hand side
x = solve(A, b)

def visualize():
    from skfem.visuals.matplotlib import plot
    return plot(m, x, shading='gouraud', colorbar=True)

if __name__ == "__main__":
    visualize().show()




np.save("E:/project/fem_nn/fem_nn_partial_data_1D_elliptic/forward_data.npy",x)
np.save("E:/project/fem_nn/fem_nn_partial_data_1D_elliptic/forward_mesh.npy",m.p)