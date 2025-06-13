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

    CellA = battery.Cell("A", operating_mode="current")
    CellB = battery.Cell("B", operating_mode="current")
    CellC = battery.Cell("C", operating_mode="current")
    CellD = battery.Cell("D", operating_mode="current")

    circuit = [["A","C"], ["B","D"]]

    DN_model = FoKL.FoKLRoutines.load("DN_model.fokl")
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
    # def DN_func(sto, T):
    #     beta_mean = parameter_values["Betas"]
    #     return beta_mean*1e-14
        # return np.exp(evaluate_pybamm(beta_mean, mtx, [sto, (T - T_min_liion) / (T_max_liion - T_min_liion)]))
    CellA.edit_params(["Current function [A]"], ["[input]"])
    CellB.edit_params(["Current function [A]"], ["[input]"])
    CellC.edit_params(["Current function [A]"], ["[input]"])
    CellD.edit_params(["Current function [A]"], ["[input]"])
    CellA.edit_params(["Ambient temperature [K]"], [328.15])
    CellB.edit_params(["Ambient temperature [K]"], [348.15])
    CellC.edit_params(["Ambient temperature [K]"], [288.15])
    CellD.edit_params(["Ambient temperature [K]"], [308.15])
    # Cell3.edit_params(["Current function [A]", "Negative particle diffusivity [m2.s-1]"], [current_func, DN_func])
    CellA.inputs = {"Current function [A]": 0.5}
    CellB.inputs = {"Current function [A]": 0.5}
    CellC.inputs = {"Current function [A]": 0.5}
    CellD.inputs = {"Current function [A]": 0.5}
    cells = [CellA, CellB, CellC, CellD]

    bat = battery.Battery(cells, [], circuit = circuit)
    bat.power_func = lambda t: 9-t/120
    t1 = timeit.default_timer()
    sol = bat.simulate([0.0, 360.0], 2)
    t2 = timeit.default_timer() - t1
    print(t2)

    bat.plot()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
