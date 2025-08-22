import timeit

import pybamm
import numpy as np
import matplotlib.pyplot as plt
from Battery.Circuit_Solvers import Circuit
import gc
from Battery.Thermal_Model import ThermalSolver
import psutil
import time

operating_dict = ["power", "voltage", "current"]
cell_dict_base = {"initial soc": 1, "param": "Chen2020", "operating mode": "current"}
def _unpack_cell_dict(dict):
    cell_dict_updated = cell_dict_base.copy()
    for val in dict:
        cell_dict_updated.update({val: dict[val]})

    initial_soc = cell_dict_updated["initial soc"]
    param = cell_dict_updated["param"]
    operating_mode = cell_dict_updated["operating mode"]
    return initial_soc, param, operating_mode

class Battery:
    def __init__(self, cells = None, nodes = None, circuit = None, create_from_circuit=False, cell_dict = None):
        if cell_dict is None:
            cell_dict = cell_dict_base
        if create_from_circuit:
            self.cells = self.create_circuit_from_string(circuit, cell_dict)
        else:
            self.cells = cells
        self.nodes = nodes
        self.batteryCount = len(self.cells)
        sols = []
        I = []
        V =[]
        T =[]
        P = []
        for i in range(len(self.cells)):
            sols.append([])
            I.append([])
            V.append([])
            T.append([])
            P.append([])
        self.solutions = sols
        self.I = I
        self.V = V
        self.T = T
        self.P = P
        self.initial_soc = 1
        executor = self._enter_pool()
        self.Circuit = Circuit(self.cells, executor, use_ids = True )
        self.solution_vars = ["Current [A]", "Voltage [V]", "Surface temperature [K]", "Power [W]"]

        # Thermal Init
        thermal_solver = ThermalSolver(self.cells,{'c':1000, 'k':0.5, 'rho':1000},np.array([2,2,1]),[], np.array([0.2, 0.4]))
        thermal_solver.set_pack_geometry(np.array([5, 2, 2]), [], np.array([0.5, 0.5, 1]))
        thermal_solver.create_mesh(0.025)
        thermal_solver._place_cell_in_pack()
        thermal_solver.cell_to_mesh()
        self.thermal_solver = thermal_solver

    def _enter_pool(self, num_workers = 8):
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=num_workers)

        return executor

    def create_circuit_from_string(self, circuit, cell_dict):
        next=False
        num_vec = []
        s = 0
        p = 0
        initial_soc, param, operating_mode = _unpack_cell_dict(cell_dict)
        for val in circuit:

            if val.isnumeric():
               num_vec.append(int(val))

            elif val == 's':
                num = sum(num_vec[-(i+1)] * 10 ** i for i in range(len(num_vec)))
                s = num
                num_vec = []
            elif val == 'p':
                num = sum(num_vec[-(i+1)] * 10 ** i for i in range(len(num_vec)))
                p = num
                num_vec = []
            else:
                raise ValueError(f"Circuit String: {circuit} must be in format XsXp")
        printed_circuit = []
        k=0
        x = 65
        for par in range(p):
            printed_circuit.append([])
            for ser in range(s):
                printed_circuit[par].append(Cell(initial_soc = initial_soc, param=param, operating_mode=operating_mode))
                printed_circuit[par][ser].cell_id = str(par) + str(ser)

        return printed_circuit
    def create_simulations(self, **kwargs):
        sims = []
        for n_branch in range(len(self.cells)):
            sims.append([])
            for i, cell in enumerate(self.cells[n_branch]):
                # cell.params.update({"Ambient temperature [K]": pybamm.InputParameter("Input temperature [K]")},
                #                         check_already_exists=False)
                sim_i = pybamm.Simulation(cell.model, parameter_values=cell.params, solver=cell.solver)
                sim_i.build(initial_soc=self.initial_soc)
                sims[n_branch].append(sim_i)
        self.sims = sims
        return sims


    def simulate(self, t_vec, dt, dt_circuit = None, current=None, voltage=None, power=None):
        if dt_circuit is None:
            dt_circuit = dt

        t_initial = t_vec[0]
        t_final = t_vec[1]
        t = t_initial
        self.t_solver = t

        self.create_simulations()

        event_hit = False
        solution_t = None
        power_t = []
        dIdt = None
        I_circuit = None
        current_func = None

        # executor = self._enter_pool()
        #
        T_now = (np.ones([200,80,80]))*273
        self.thermal_solver.dt = dt

        while t < t_final:
            if t == t_initial:
                solution_t_i = self._first_step(dt)
            else:
                solution_t_i = self._step_body(dt, solution_t, t)
            for n_branch in range(len(solution_t_i)):
                for i, sol in enumerate(solution_t_i[n_branch]):
                    if sol.termination != 'final time':
                        event_hit = True
                        termination_event = sol.termination
                        termination_cell = i + 1
                        sol_event = sol
                        t_event = sol_event.t_event
            # T_t = self.update_params(solution_cells_t)
            if event_hit:
                dt_adjusted = t_event - t
                try:
                    solution_t_i_end = self._step_body(dt_adjusted, solution_t)
                except:
                    raise TypeError(f"Battery conditions failed on initial time step on cell {termination_cell} with {termination_event} ")
                break
            else:
                if np.mod(t, dt_circuit) == 0:
                    self.Circuit.P_goal = self.power_func(t)

                    I_circuit, P_tot, sol_new, function_params, P_vec = self.Circuit.solve(self.sims, self.cells, dt_circuit, solution_t_i, t, I_circuit, dIdt, old_current_func = current_func)


                    # print(I_circuit)
                    current_func = [[],[]]
                    current_func[0] = self._create_current_poly(function_params, 0)
                    current_func[1] = self._create_current_poly(function_params, 1)

                k=0
                n_cells = sum(np.shape(self.cells))
                if current_func is not None:
                    for i in range(len(self.cells)):
                        for j in range(len(self.cells[i])):
                            WpC = (P_vec[i][j]['Power [W]'] / n_cells) * np.ones(n_cells)
                            Q_cell = self.thermal_solver.calculate_Q_dot(WpC)
                            Q_compiled = self.thermal_solver.compile_Q_dot(Q_cell)

                            T_next = self.thermal_solver.heat_equation_full(T_now, Q_compiled)
                            T_next_pad = np.pad(T_next, 1, constant_values=225)
                            self.thermal_solver.update_average_temperature(T_next_pad)
                            self.cells[i][j].inputs.update({'Current function [A]': current_func[i](t)})
                            self.cells[i][j].inputs.update({'Ambient temperature [K]': self.cells[i][j].average_temperature})
                            k+=1
                if solution_t:
                    T_now = T_next_pad
                    solution_t_i = self._step_body(dt, solution_t, t)
                    # solution_t_i = sol_new
                solution_t = solution_t_i
            print(t)

            t += dt
            self.t_solver = t

            gc.collect()
        self.final_sol = solution_t
        # solution_cells = self._pull_solution(solution_t)

        return solution_t

    def _set_for_plot(self, solution_cells):
        V_cells = []
        VV_cell = []
        I_cells = []
        T_cells = []
        power_t = []
        t = []
        P_pack = 0
        V_pack = 0
        V_branch = np.zeros([len(solution_cells[0][0]["Voltage [V]"].entries), len(solution_cells)]).T

        for n_branch in range(len(solution_cells)):
            for i, cell in enumerate(solution_cells[n_branch]):
                t_cell = cell["Time [s]"].entries
                V_cell = cell["Voltage [V]"].entries
                # nn = min([len(V_branch), len(V_cell)])
                V_branch[n_branch] += V_cell
                I_cells.append(cell["Current [A]"].entries)
                T_cells.append(cell["Surface temperature [K]"].entries)
                P_pack += cell["Power [W]"].entries
        self.P_pack = P_pack
        for t_i in solution_cells[0][0]['Time [s]'].entries:
            power_t.append(self.power_func(t_i))
        self.power_t = np.array(power_t)

        return V_branch, I_cells, T_cells, P_pack



    def _pull_solution(self, solution_t, solution_vars=None):
        if solution_vars is None:
            solution_vars = self.solution_vars
        solution_dict = {"Time [s]": solution_t[0]["Time [s]"].entries}
        solution_cells = []
        for i in range(len(self.cells)):
            solution_cells.append(solution_dict.copy())
            for solution in solution_vars:
                solution_cells[i].update({solution: np.array(solution_t[i][solution].entries)})
        return solution_cells


    def _first_step(self, dt):
        solution_t = []
        for n_branch in range(len(self.sims)):
            solution_t.append([])
            for i, sim in enumerate(self.sims[n_branch]):
                solution_i = sim.step(dt, starting_solution=None, inputs = self.cells[n_branch][i].inputs, calculate_sensitivities=False,t_interp = np.linspace(self.t_solver, dt + self.t_solver, 2))#, inputs={"Input temperature [K]": self.inputs[i]})
                solution_t[n_branch].append(solution_i)
        return solution_t

    def _step_body(self, dt, solution_tm1, t, start=False):
        solution_t = []
        for n_branch in range(len(self.cells)):
            solution_t.append([])
            for i, sim in enumerate(self.sims[n_branch]):
                solution_i = solution_tm1[n_branch][i]
                solution_ip1 = sim.step(dt, starting_solution=solution_i, inputs = self.cells[n_branch][i].inputs, calculate_sensitivities=False, t_interp = np.linspace(0, dt, 2))#, inputs={"Input temperature [K]": self.inputs[i]})
                solution_t[n_branch].append(solution_ip1)
        return solution_t

    def _unroll_sims(self, sims, cells, solution_tm1):
        sims_unrolled = []
        cells_unrolled = []
        solution_unrolled = []

        k_vec = []
        k = 0
        for n_branch in range(len(sims)):
            for i in range(len(sims[n_branch])):
                sims_unrolled.append(sims[n_branch][i])
                cells_unrolled.append(cells[n_branch][i])
                solution_unrolled.append(solution_tm1[n_branch][i])
                k_vec.append([k, n_branch, i])
                k += 1

        return sims_unrolled, cells_unrolled, solution_unrolled, k_vec


    def _roll_sims(self, sims_unrolled, cells_unrolled, solution_unrolled, k_vec):
        k_max, n_branch_max, i_max = np.max(k_vec, axis=0)

        sims_rolled = []
        cells_rolled = []
        solution_rolled = []

        k = 0

        for n_branch in range(n_branch_max + 1):
            sims_rolled.append([])
            cells_rolled.append([])
            solution_rolled.append([])

            for i in range(i_max + 1):
                sims_rolled[n_branch].append(sims_unrolled[k])
                cells_rolled[n_branch].append(cells_unrolled[k])
                solution_rolled[n_branch].append(solution_unrolled[k])

                k += 1

        return sims_rolled, cells_rolled, solution_rolled

    def _sim_step_par(self, args):
        sim, cell, dt, sol, input = args
        test_input = cell.inputs.copy()
        test_input.update({'Current function [A]': input})

        sol_np1 = sim.step(dt, starting_solution=sol, inputs=test_input)

    def _step_body_p(self, executor, dt, solution_tm1, t, num_workers=6):
        # def _step_body(self, dt, solution_tm1, t, start=False):
        #     solution_t = []
        #     for n_branch in range(len(self.cells)):
        #         solution_t.append([])
        #         for i, sim in enumerate(self.sims[n_branch]):
        #             solution_i = solution_tm1[n_branch][i]
        #             solution_ip1 = sim.step(dt, starting_solution=solution_i, inputs=self.cells[n_branch][i].inputs,
        #                                     t_interp=np.linspace(0, dt,
        #                                                          4))  # , inputs={"Input temperature [K]": self.inputs[i]})
        #             solution_t[n_branch].append(solution_ip1)
        #     return solution_t

        def _sim_step_par(args):
            sim, cell, dt, sol = args
            sol_np1 = sim.step(dt, starting_solution=sol, inputs=cell.inputs, calculate_sensitivities=False)
            return sol_np1

        sims_ur, cells_ur, sols_ur, k_vec = self._unroll_sims(self.sims, self.cells, solution_tm1)

        futures = [executor.submit(_sim_step_par, (sim, cell, dt, sol))
                   for sim, cell, sol in zip(sims_ur, cells_ur, sols_ur)]

        sol_p1 = [f.result() for f in futures]

        null, null, sol_rolled = self._roll_sims(sims_ur, cells_ur, sol_p1, k_vec)

        return sol_rolled

    def update_return_solutions(self, solution_var):
        self.solution_vars = [solution_var]

    def update_params(self, cell_temperatures):
        T_t = []
        for i, cell in enumerate(self.cells):
            T_t_i = cell_temperatures[i]['Surface temperature [K]'][-1:][0]
            T_t.append(T_t_i)
        return T_t
    #
    # def _current_to_circuit(self, I_new):
    #     for i in range(self.Circuit.n_branch):
    def plot(self):
        self.t = self.final_sol[0][0]['Time [s]'].entries
        V_branch, I_cells, T_cells, P_pack = self._set_for_plot(self.final_sol)
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        # ax1.plot(self.t[0][:len(self.V_pack)], self.V_pack, label = "Pack Voltage")
        for cell_i, V_cell in enumerate(V_branch):
            ax1.plot(self.t, V_cell, label=f"V Branch {cell_i}")
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Voltage Branch [V]')
        # ax1_pack = ax1.twinx()
        # ax1_pack.plot(self.t[0], self.V_pack)
        # ax1_pack.set_xlabel('Time (s)')
        # ax1_pack.set_ylabel('Voltage Pack [V]')

        # ax1.plot(self.t[2], self.V_cells[0] + self.V_cells[1], label="V Branch 1")
        # ax1.plot(self.t[2], self.V_cells[2] + self.V_cells[3], label="V Branch 2")
        ax1.set_title("Pack Voltage")
        ax1.legend()

        ax2.plot(self.t, I_cells[0], label="I Branch 1")
        ax2.plot(self.t, I_cells[2], label="I Branch 2")
        ax2.plot(self.t, I_cells[0] + I_cells[2], label="I Pack")
        ax2.set_title("Pack Current")
        ax2.legend()

        ax3.plot(self.t, T_cells[0], label="T Branch 1")
        ax3.plot(self.t, T_cells[2], label="T Branch 2")
        # ax3.plot(self.t[2], self.T_cells[2], label="T Cell 3")
        ax3.set_title("Pack Temperature")
        ax3.legend()

        ax4.plot(self.t, self.power_t, label="Power goal")
        ax4.plot(self.t, P_pack, label = "Power Output")

        ax4.set_title("Pack Power Goal")
        ax4.legend()
        plt.show()

    def _create_current_poly(self,function_params, i):
        t_p_val = float(function_params["Time Now"])
        I_val = float(function_params["Current"][i])
        dIdt_val = float(function_params["dCurrent"][i])
        dIIdtt_val = float(function_params["ddCurrent"][i])

        def current_func(t):
            current = I_val + (dIdt_val * (t - t_p_val)) + ((dIIdtt_val * (t - t_p_val) ** 2) / 2)
            return current

        return current_func

    def edit_params(self, function_dict, gp_structure):
        for cell in self.cells:
            cell.edit_params(function_dict, gp_structure)


class Cell:
    def __init__(self, initial_soc=1, param="Marquis2019", operating_mode="current", cell_id = None):
        if operating_mode not in operating_dict:
            raise NotImplementedError(f"Operating mode {operating_mode} not available. Must be one of:'power', 'voltage', or 'current' ")
        self.initial_soc = initial_soc
        self.solver = pybamm.IDAKLUSolver()
        self.param_name = param
        self.create_params()
        self.create_battery(operating_mode=operating_mode)
        if cell_id is None:
            self.cell_id = id(self)
        else:
            self.cell_id = cell_id

    def edit_params(self, function_dict, gp_structure):
        if self.params is None:
            param = pybamm.ParameterValues(self.param_name)
        else:
            param = self.params
        for i, parameter_function in enumerate(function_dict):
            param[parameter_function] = gp_structure[i]
        self.params = param
        return param

    def create_new_params(self, str, value, check_already_exists=False):
        self.params.update({str: value}, check_already_exists=check_already_exists)
    def create_params(self):
        self.params = pybamm.ParameterValues(self.param_name)

    def create_battery(self, operating_mode="current"):
        options = {"operating mode": operating_mode, "thermal": "x-full"}
        self.operating_mode = operating_mode
        model = pybamm.lithium_ion.SPM(options=options)
        self.model = model
