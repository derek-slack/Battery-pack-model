from Battery.Thermal_Model import ThermalSolver
from Battery.battery import Battery
from Battery.Circuit_Solvers import Circuit

import numpy as np
import matplotlib
matplotlib.use("QtAgg")
import matplotlib.pyplot as plt


bat = Battery(circuit="2s2p",create_from_circuit=True)

ts = ThermalSolver(bat.cells,{'c':1000, 'k':0.5, 'rho':1000},np.array([2,2,1]),[], np.array([0.2, 0.4]))
ts.set_pack_geometry(np.array([5,2,2]), [],np.array([0.5,0.5,1]))
mesh  = ts.create_mesh(0.025)
ts._place_cell_in_pack()
ts.cell_to_mesh([])


W = 10
n_cells = 4
WpC = (W/n_cells)*np.ones(n_cells)
Q_cell = ts.calculate_Q_dot(WpC)
Q_compiled = ts.compile_Q_dot(Q_cell)

Tg = (np.ones([200,80,80]))*273

ig = np.linspace(0,24,25)
jg = np.linspace(0,24,25)
kg = np.linspace(0,24,25)


ts.dt = 1
for t in range(0,500):
    T_next = ts.heat_equation_full(Tg, Q_compiled)
    Tg = np.pad(T_next,1, constant_values=225)
    print(np.average(Tg))


h=1