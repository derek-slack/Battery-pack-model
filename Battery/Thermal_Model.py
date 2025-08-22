import timeit

import numpy as np
from itertools import product
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt
class ThermalSolver():
    def __init__(self, cells, heat_params, cell_n_xyz, cell_ids, cell_rz):

        self.cell_n_xyz = cell_n_xyz # Number of cells in x, y, z direction geometrically i.e:
        # [4, 2, 1] - 4 rows of 2 cells in 1 cell width
        self.cell_id = []
        self.cells = cells
        self.cell_rz = cell_rz
        self.heat_params = heat_params
        # self.num_cells = sum(len(branch) for branch in cells)
        # self.num_rows = rows_cells
        # self.num_cols = np.floor(self.num_cells/self.num_rows)
        # self.extra = np.mod(self.num_cells, self.num_rows)
        # if self.extra == 0:
        #     self.even_pack = True
        # else:
        #     self.even_pack = False

    def edit_heat_params(self, heat_param, update_val):
        '''
        Updates thermal parameter set values
        :param heat_param: parameter/s to update
        :param update_val: values to update parameter to
        :return: updated parameter set
        '''
        update_params = self.heat_params
        for param_str, param_val in zip(heat_param, update_val):
            update_params[param_str] = param_val
        self.heat_params = update_params
        return update_params

    def set_pack_geometry(self, pack_xyz, boundary_conditions, cell_to_wall_xyz):
        self.pack_xyz = pack_xyz
        self.cell_to_wall_xyz = cell_to_wall_xyz
        self.boundary_conditions = boundary_conditions

        self.pack_width = pack_xyz[0]
        self.pack_length = pack_xyz[1]
        self.pack_height = pack_xyz[2]

        print(f"Pack created: \n Width (m): {self.pack_width} \n Length (m): {self.pack_length} \n Height (m): {self.pack_height}")

    def create_mesh(self, mesh_xyz, method = "dx", regular_mesh = True):
        mesh_created = False

        if method == "dx":
            if regular_mesh:
                dx = mesh_xyz
                dy = dx
                dz = dx
            else:
                dx = mesh_xyz[0]
                dy = mesh_xyz[1]
                dz = mesh_xyz[2]

        elif method == "number_cells":
            if regular_mesh:
                dx = self.pack_width/mesh_xyz
                dy = dx
                dz = dx
            else:
                dx = self.pack_width/mesh_xyz[0]
                dy = self.pack_length/mesh_xyz[1]
                dz = self.pack_height/mesh_xyz[2]

        elif method == "explicit":
            dx = mesh_xyz[0][1] - mesh_xyz[0][0]
            dy = mesh_xyz[1][1] - mesh_xyz[1][0]
            dz = mesh_xyz[2][1] - mesh_xyz[2][0]
            mesh_created = True

        self.dx = dx
        self.dy = dy
        self.dz = dz

        if mesh_created:
            mesh = mesh_xyz
        else:
            nx = np.ceil(self.pack_width/dx).astype(int)
            ny = np.ceil(self.pack_length/dy).astype(int)
            nz = np.ceil(self.pack_height/dz).astype(int)

            mesh = np.zeros([nx,ny,nz,3])

            for i in range(nx):
                for j in range(ny):
                    for k in range(nz):
                        mesh[i, j, k, :] = np.array([i*dx, j*dy, k*dz])

        self.mesh = ThermalMesh(mesh)

        self.mesh_x = np.linspace(0, self.pack_width, nx)
        self.mesh_y = np.linspace(0, self.pack_length, ny)
        self.mesh_z = np.linspace(0, self.pack_height, nz)

        return mesh

    def _place_cell_in_pack(self):
        # space_between_cells = (self.pack_xyz - 2*self.cell_to_wall_xyz)/(self.cell_n_xyz - np.ones(3))
        cell_pos = []
        for dir in range(3):
            cell_pos.append(np.linspace(self.cell_to_wall_xyz[dir], (self.pack_xyz[dir]-self.cell_to_wall_xyz[dir]), self.cell_n_xyz[dir]))

        cell_pos_combs = list(product(cell_pos[0], cell_pos[1], cell_pos[2]))

        count_cell = 0

        for branch in self.cells:
            for cell in branch:
                cell.position = np.array((list(cell_pos_combs[count_cell]))).reshape(-1,1)
                cell.count = count_cell
                count_cell += 1

    def cell_to_mesh(self):

        for branch in self.cells:
            for cell in branch:

                # to centers
                dx_c = self.mesh.mesh_points[:, :, :, 0] - cell.position[0]
                dy_c = self.mesh.mesh_points[:, :, :, 1] - cell.position[1]
                dz_c = self.mesh.mesh_points[:, :, :, 2] - cell.position[2]
                dr_c = np.sqrt(dx_c**2 + dy_c**2)

                # dist to bound
                dr_b = dr_c - self.cell_rz[0]  # distance to border
                dz_b = (abs(dz_c) - (self.cell_rz[1])/2)  # distance to border


                d_gen = 0.075 # Generation term

                # Outer side surface
                side_wall = (abs(dr_b) <= d_gen) & (abs(dz_c) <= self.cell_rz[1] / 2)

                # Top and bottom caps
                end_caps = (abs(dz_b) <= d_gen) & (dr_c <= self.cell_rz[0])

                # Combine
                all_gen = side_wall | end_caps
                all_bound = ((dr_c < self.cell_rz[0]) & (abs(dz_c) < self.cell_rz[1] / 2)) & (~all_gen)
                all_active = (~all_gen) & (~all_bound)

                cell.generation_cells = all_gen
                cell.bound_cells = all_bound
                cell.all_active = all_active

                self.mesh.generation_cells |= all_gen
                self.mesh.boundary_cell |= all_bound
                self.mesh.active |= all_active

    def calculate_Q_dot(self, Q_dot_model):
        count_cell = 0
        Q_dot_cell = []
        for branch in self.cells:
            for cell in branch:
                Q_dot_cell.append(Q_dot_model[count_cell]/sum(sum(sum(cell.generation_cells))))
                cell.q_dot_t = Q_dot_cell[count_cell]
                count_cell += 1
        return Q_dot_cell

    def Q_dot_to_matrix(self, Q_dot_cell, cell):
        return (Q_dot_cell/(self.dx*self.dy*self.dz))*(cell.generation_cells + np.zeros(self.mesh.mesh_points[:,:,:,0].shape))

    def compile_Q_dot(self, Q_dot_cell):
        Q_dot_mesh = np.zeros(self.mesh.mesh_points[:, :, :, 0].shape)
        k=0
        for branch in self.cells:
            for cell in branch:
                Q_dot_mesh += self.Q_dot_to_matrix(Q_dot_cell[k],cell)
                k+=1
        return Q_dot_mesh

    def laplacian_bc_matrix(self):
        pass

    def _print_volume(self):
        pass

    def heat_equation_full(self, T, Q_dot):

        k = self.heat_params['k']
        rho = self.heat_params['rho']
        c = self.heat_params['c']

        # Shifted slices
        T_im1 = T[:-2, 1:-1, 1:-1]
        T_ip1 = T[2:, 1:-1, 1:-1]
        T_jm1 = T[1:-1, :-2, 1:-1]
        T_jp1 = T[1:-1, 2:, 1:-1]
        T_km1 = T[1:-1, 1:-1, :-2]
        T_kp1 = T[1:-1, 1:-1, 2:]

        T_center = T[1:-1, 1:-1, 1:-1]
        Q_dot_center = Q_dot[1:-1, 1:-1, 1:-1]


        d2T_dx2 = (T_ip1 - 2 * T_center + T_im1) / (self.dx ** 2)
        d2T_dy2 = (T_jp1 - 2 * T_center + T_jm1) / (self.dy ** 2)
        d2T_dz2 = (T_kp1 - 2 * T_center + T_km1) / (self.dz ** 2)

        laplacian = (d2T_dx2 + d2T_dy2 + d2T_dz2)*(~self.mesh.boundary_cell[1:-1, 1:-1, 1:-1])

        T_next = T_center + (k / (rho * c)) * laplacian * self.dt + (Q_dot_center * self.dt) / (rho * c)


        return T_next

    def create_heat_vmap(self):
        pass

    def update_average_temperature(self, T_next):
        for branch in self.cells:
            for cell in branch:
                average_temperature = np.mean(cell.generation_cells * T_next)
                cell.average_temperature = average_temperature

class ThermalMesh:
    def __init__(self, mesh_points):
        self.mesh_points = mesh_points
        shape = self.mesh_points.shape[:3]  # (Nx, Ny, Nz)

        self.generation_cells = np.zeros(shape, dtype=bool)
        self.boundary_cell = np.zeros(shape, dtype=bool)
        self.active = np.zeros(shape, dtype=bool)

