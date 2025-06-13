from Battery.new_eval import *
import FoKL
from FoKL import FoKLRoutines
import timeit
model = pybamm.lithium_ion.SPM()
solver = pybamm.IDAKLUSolver(options={'num_threads': 4})

DN_model = FoKL.FoKLRoutines.load("DN_model.fokl")
betas = DN_model.betas
mtx = DN_model.mtx
bm = np.mean(betas, axis=0).reshape(1, -1)

T_max_liion = 80 + 273.15
T_min_liion = -30 + 273.15

params = pybamm.ParameterValues("Chen2020")
BB = pybamm.InputParameter("Betas", expected_size=16)

params.update({"Betas": BB},check_already_exists=False)
params.update({"mtx": mtx},check_already_exists=False)

# m = evaluate_pybamm_clone(bm,mtx,[0.5, 0.5])


# params["Current function [A]"] = "[input]"
#
#
# params.update(
#     {
#         "Current function [A]": "[input]",
#     }
# )

n = 1e3
current_inputs = [
    {"Current function [A]": current} for current in np.linspace(0, 0.6, int(n))
]

current_inputs_custom = [{"Current function [A]": 1.}, {"Current function [A]": 0.9}, {"Current function [A]": 0.8}]

num_threads_list = [1, 2, 4, 8, 16, 32]

# model = pybamm.lithium_ion.SPM()
# params = pybamm.ParameterValues("Chen2020")
# params.update(
#     {
#         "Current function [A]": "[input]",
#     }
# )

def DP_func(sto, T):
    betas = params["Betas"]
    mtx = params["mtx"]
    # return 1e-14
    res = np.exp(evaluate_pybamm(betas, mtx, [sto, (T - T_min_liion) / (T_max_liion - T_min_liion)]))
    return res
params["Positive particle diffusivity [m2.s-1]"] = DP_func

solver = pybamm.IDAKLUSolver(options={"num_threads": 8})
sim = pybamm.Simulation(model, solver=solver, parameter_values=params)
start_time = timeit.default_timer()
for i in range(32):
    sol = sim.solve([0, 3600], inputs={"Betas": bm[0]})

end_time = timeit.default_timer()
print(
    f"Time taken to solve 1000 SPM simulation for {8} threads: {end_time - start_time:.2f} s"
)


