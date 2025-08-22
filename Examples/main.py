# This is a sample Python script.
import timeit

import FoKL.FoKLRoutines

from Battery import battery
import matplotlib
matplotlib.use('TkAgg')

from Battery.new_eval import *
# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    CellA = battery.Cell(cell_id="A", operating_mode="current")
    CellB = battery.Cell(cell_id="B", operating_mode="current")
    CellC = battery.Cell(cell_id="C", operating_mode="current")
    CellD = battery.Cell(cell_id="D", operating_mode="current")

    DN_model = FoKL.FoKLRoutines.load("Examples/DN_model.fokl")
    betas = DN_model.betas
    mtx = DN_model.mtx

    T_max_liion = 80 + 273.15
    T_min_liion = -30 + 273.15

    def current_func(t):
        return 1.5*np.sin(t/100) + 0.75

    def power_func(t):
        return -1.05 * t + 5
    # bm = np.mean(betas, axis=0).reshape(1,-1)
    #
    # parameter_values = pybamm.ParameterValues("Chen2020")
    # parameter_values.update({"Betas": "[input]"}, check_already_exists=False)


    # parameter_values.update({"mtx": [1]},check_already_exists=False)
    # Cell1.create_new_params("Betas", "[input]")
    # Cell2.create_new_params("Betas","[input]")
    def DN_func(sto, T):
        beta_mean = betas

        return np.exp(evaluate_pybamm_clone(beta_mean, mtx, [sto, (T - T_min_liion) / (T_max_liion - T_min_liion)]))



    CellA.edit_params(["Current function [A]"], ["[input]"])
    CellB.edit_params(["Current function [A]"], ["[input]"])
    CellC.edit_params(["Current function [A]"], ["[input]"])
    CellD.edit_params(["Current function [A]"], ["[input]"])
    # CellA.edit_params(["Ambient temperature [K]"], ["[input]"])
    # CellB.edit_params(["Ambient temperature [K]"], ["[input]"])
    # CellC.edit_params(["Ambient temperature [K]"], ["[input]"])
    # CellD.edit_params(["Ambient temperature [K]"], ["[input]"])
    # CellA.edit_params(["Negative particle diffusivity [m2.s-1]"], [DN_func])
    CellA.inputs = {"Current function [A]": 1.51} #, "Ambient temperature [K]": 299.15}
    CellB.inputs = {"Current function [A]": 1.51} #, "Ambient temperature [K]": 299.15}
    CellC.inputs = {"Current function [A]": 1.51} #, "Ambient temperature [K]": 299.15}
    CellD.inputs = {"Current function [A]": 1.51} #, "Ambient temperature [K]": 299.15}
    cells = [[CellA, CellB], [CellC, CellD]]

    def power_t(t):
        if t <= 121:
            p=15
        elif t <= 1501:
            p = 10 + (-3/1380)*(t-120)
        else:
            p = 12
        return p
    bat = battery.Battery(cells, [])
    bat.power_func = power_t
    t1 = timeit.default_timer()
    sol = bat.simulate([0.0, 1800], 5, 5)
    t2 = timeit.default_timer() - t1
    print(t2)

    bat.plot()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
