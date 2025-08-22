import timeit

from jax import jit
from pathos.pools import ProcessPool
import pybamm
import numpy as np
import jax



model = pybamm.lithium_ion.SPM()
solver1 = pybamm.IDAKLUSolver()
solver2 = pybamm.IDAKLUSolver()
solver3 = pybamm.IDAKLUSolver()
solver4 = pybamm.IDAKLUSolver()
solver5 = pybamm.IDAKLUSolver()
solver6 = pybamm.IDAKLUSolver()
solver7 = pybamm.IDAKLUSolver()
solver8 = pybamm.IDAKLUSolver()


param = pybamm.ParameterValues("Chen2020")
param['Current function [A]'] = "[input]"

sim = pybamm.Simulation(model, parameter_values=param, solver=solver1)
sim2 = pybamm.Simulation(model, parameter_values=param, solver=solver2)
sim3 = pybamm.Simulation(model, parameter_values=param, solver=solver3)
sim4 = pybamm.Simulation(model, parameter_values=param, solver=solver4)
sim5 = pybamm.Simulation(model, parameter_values=param, solver=solver5)
sim6 = pybamm.Simulation(model, parameter_values=param, solver=solver6)
sim7 = pybamm.Simulation(model, parameter_values=param, solver=solver7)
sim8 = pybamm.Simulation(model, parameter_values=param, solver=solver8)
t = timeit.default_timer()

sims11 = [sim, sim2, sim3, sim4, sim5, sim6, sim7, sim8]
sims22 = sims11.copy()

inds = np.array(range(len(sims11)))

class SimHolder:
    def __init__(self, sims):
        self.sims = sims
        s = []
        for i in range(len(sims)):
            s.append(None)
        self.sol = s

    def step(self, i):
        sol = self.sims[i].step(1, starting_solution = self.sol[i])
        self.sol[i] = sol
        return sol

simC2 = SimHolder(sims11)
simC1 = SimHolder(sims22)

time = timeit.default_timer() - t
print(time)
t2 = timeit.default_timer()
#
# def run_simulation_step(args):
#     sim, sol = args
#     return sim.step(1, starting_solution=sol)
#
# with ProcessPool() as pool:
#     args = [(sim, sol) for sim, sol in zip(sims11, simC2.sol)]
#     results = pool.map(run_simulation_step, args)

from concurrent.futures import ThreadPoolExecutor

def run_single_step_thread(args):
    sim, sol = args
    return sim.step(1, inputs = {'Current function [A]': 1}, starting_solution=sol, calculate_sensitivities=True)


# Create persistent thread pool
executor = ThreadPoolExecutor(max_workers=8)

for step in range(720):


    futures = [executor.submit(run_single_step_thread, (sim, sol))
               for sim, sol in zip(sims11, simC2.sol)]

    simC2.sol = [f.result() for f in futures]

    h=12

executor.shutdown()

time_second = timeit.default_timer() - t2
print(time_second)
# time1 = timeit.default_timer()
# for step in range(720):
#     for i in range(8):
#         simC1.step(i)
# print(timeit.default_timer() - time1)
h=1