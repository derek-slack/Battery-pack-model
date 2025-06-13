import pybamm
import numpy as np

def DP_1(sto, T):
    return sto*1e-15

def DP_2(sto, T):
    return sto*1e-14


model = pybamm.lithium_ion.SPM()
solver = pybamm.IDAKLUSolver()

param = pybamm.ParameterValues("Chen2020")

param.update({"Negative particle diffusivity [m2.s-1]": "[input]"})


sim = pybamm.Simulation(model, parameter_values=param, solver=solver)

sol = sim.solve([0, 3600], inputs=[{"Negative particle diffusivity [m2.s-1]": DP_1}, {"Negative particle diffusivity [m2.s-1]": DP_1}])