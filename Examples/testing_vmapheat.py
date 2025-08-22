import jax.numpy as jnp
import numpy as np
import jax
from Battery.Thermal_Model import ThermalSolver
np.set_printoptions(legacy='1.25')

Tg = np.ones([25,25,25]) * np.random.normal(np.ones([25,25,25]), 0.75)

ig = np.linspace(0,24,25)
jg = np.linspace(0,24,25)
kg = np.linspace(0,24,25)

rowg = 0.75
cg = 2
kg = 0.75

Q_dotg = np.zeros([25,25,25])


h=1
# def heat_equation(T, i, j, k, row, c, Q_dot):
#     T_t = T[i, j, k]
#
#     T_im1jk = T[i - 1, j, k]
#     T_ijm1k = T[i, j - 1, k]
#     T_ijkm1 = T[i, j, k - 1]
#
#     T_ip1jk = T[i + 1, j, k]
#     T_ijp1k = T[i, j + 1, k]
#     T_ijkp1 = T[i, j, k + 1]
#
#     Q_dot_ijk = Q_dot[i,j,k]
#
#     T_tp1 = T_t + (k / (row * c)) * (T_im1jk + T_ijm1k + T_ijkm1 - 6 * T_t - T_ip1jk - T_ijp1k - T_ijkp1) + Q_dot_ijk / (
#                 row * c)
#
#     return T_tp1

def heat_equation_full(T, rho, c, k, Q_dot):
    # Shifted slices
    T_im1 = T[:-2, 1:-1, 1:-1]
    T_ip1 = T[2:, 1:-1, 1:-1]
    T_jm1 = T[1:-1, :-2, 1:-1]
    T_jp1 = T[1:-1, 2:, 1:-1]
    T_km1 = T[1:-1, 1:-1, :-2]
    T_kp1 = T[1:-1, 1:-1, 2:]

    T_center = T[1:-1, 1:-1, 1:-1]
    Q_dot_center = Q_dot[1:-1, 1:-1, 1:-1]

    laplacian = T_im1 + T_ip1 + T_jm1 + T_jp1 + T_km1 + T_kp1 - 6 * T_center

    T_next = T_center + (k / (rho * c)) * laplacian * self.dt + (Q_dot_center*self.dt) / (rho * c)

    return T_next

def convection_boundary():
    pass


def insulated_boundary():
    pass

# T_next = heat_equation_full(Tg, rowg, cg, kg, Q_dotg)



#   def __init__(self, cells, heat_params, rows_cells):
blank = ThermalSolver([], [], [])
blank.set_pack_geometry([1,2,3], [])
mesh = blank.create_mesh(0.25, regular_mesh=True)

h=1
