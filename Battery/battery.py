import pybamm
import numpy as np
import matplotlib.pyplot as plt
from Battery.Circuit_Solvers import Circuit

operating_dict = ["power", "voltage", "current"]
class Battery:
    def __init__(self, cells, nodes, circuit):
        self.cells = cells
        self.nodes = nodes
        self.batteryCount = len(cells)
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
        self.power_func = lambda t: 9 - t/1200
        self.Circuit = Circuit(self.cells, circuit, 8)
        self.solution_vars = ["Current [A]", "Voltage [V]", "Surface temperature [K]"]

    def create_simulations(self, **kwargs):
        sims = []
        for i, cell in enumerate(self.cells):
            # cell.params.update({"Ambient temperature [K]": pybamm.InputParameter("Input temperature [K]")},
            #                         check_already_exists=False)
            sim_i = pybamm.Simulation(cell.model, parameter_values=cell.params, solver=cell.solver)
            sim_i.build(initial_soc=self.initial_soc)
            sims.append(sim_i)
        self.sims = sims
        return sims


    def simulate(self, t_vec, dt, current=None, voltage=None, power=None):
        t_initial = t_vec[0]
        t_final = t_vec[1]
        t = t_initial
        #
        # if current is not None:
        #     for i, cell in enumerate(self.cells):
        #         cell.params['Current function [A]'] = current
        # elif voltage is not None:
        #     for i, cell in enumerate(self.cells):
        #         cell.params.update({"Voltage function [V]": pybamm.InputParameter("Voltage [V]")},
        #                                 check_already_exists=False)
        #         self.voltage = voltage

        self.create_simulations()
        event_hit = False
        solution = []
        solution_t = None
        while t < t_final:
            if t == t_initial:
                solution_t_i = self._first_step(dt)
            else:
                solution_t_i = self._step_body(dt, solution_t, t)
            # for sim in self.sims:
            #     sim.parameter_values()
            # solution_cells_t = self._pull_solution(solution_t, ["Time [s]"])

            for i, sol in enumerate(solution_t_i):
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
                self.Circuit.P_goal = self.power_func(t)

                I_circuit, P_tot, sol_new = self.Circuit.solve(self.sims,self.cells,dt,solution_t_i,t)
                k=0
                # print(P_tot)
                for i in range(2):
                    for j in range(2):
                        self.cells[k].inputs.update({'Current function [A]': I_circuit[i]})
                        k+=1
                if solution_t:
                    solution_t_i = self._step_body(dt, solution_t, t)
                solution_t = solution_t_i
            print(t)
            t += dt

        solution_cells = self._pull_solution(solution_t)
        self.V_cells = []
        self.V_pack = np.zeros(len(solution_cells[0]["Voltage [V]"]))
        self.I_cells = []
        self.T_cells = []
        self.t = []

        for i,cell in enumerate(solution_cells):
            self.t.append(cell["Time [s]"])
            self.V_cells.append(cell["Voltage [V]"])
            nn = min([len(self.V_pack),len(self.V_cells[i])])
            self.V_pack = self.V_cells[i][:nn]+self.V_pack[:nn]
            self.I_cells.append(cell["Current [A]"])
            self.T_cells.append(cell["Surface temperature [K]"])

        self.power_t = self.power_func(self.t[0])
        return solution_cells

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

        for i, sim in enumerate(self.sims):
            solution_i = sim.step(dt, starting_solution=None, inputs = self.cells[i].inputs)#, inputs={"Input temperature [K]": self.inputs[i]})
            solution_t.append(solution_i)
        return solution_t

    def _step_body(self, dt, solution_tm1, t, start=False):
        solution_t = []
        for i, sim in enumerate(self.sims):
            solution_i = solution_tm1[i]
            solution_ip1 = sim.step(dt, starting_solution=solution_i, inputs = self.cells[i].inputs)#, inputs={"Input temperature [K]": self.inputs[i]})
            solution_t.append(solution_ip1)
        return solution_t

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
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4)
        # ax1.plot(self.t[0][:len(self.V_pack)], self.V_pack, label = "Pack Voltage")
        ax1.plot(self.t[0], self.V_cells[0], label="V Cell 1")
        ax1.plot(self.t[2], self.V_cells[1], label="V Cell 2")
        ax1.plot(self.t[2], self.V_cells[2], label="V Cell 3")
        ax1.plot(self.t[2], self.V_cells[3], label="V Cell 4")
        ax1.plot(self.t[2], self.V_cells[0] + self.V_cells[1], label="V Branch 1")
        ax1.plot(self.t[2], self.V_cells[2] + self.V_cells[3], label="V Branch 2")
        ax1.set_title("Pack Voltage")
        ax1.legend()

        ax2.plot(self.t[0], self.I_cells[0], label="I Branch 1")
        ax2.plot(self.t[2], self.I_cells[2], label="I Branch 2")
        ax2.plot(self.t[2], self.I_cells[0] + self.I_cells[2], label="I Pack")
        ax2.set_title("Pack Current")
        ax2.legend()

        ax3.plot(self.t[0], self.T_cells[0], label="T Branch 1")
        ax3.plot(self.t[2], self.T_cells[2], label="T Branch 2")
        # ax3.plot(self.t[2], self.T_cells[2], label="T Cell 3")
        ax3.set_title("Pack Temperature")
        ax3.legend()

        ax4.plot(self.t[0], self.power_t, label="Power goal")
        ax4.set_title("Pack Power Goal")
        ax4.legend()
        plt.show()



class Cell:
    def __init__(self, cell_id, initial_soc=1, param="Chen2020", operating_mode="power"):
        if operating_mode not in operating_dict:
            raise NotImplementedError(f"Operating mode {operating_mode} not available. Must be one of:'power', 'voltage', or 'current' ")
        self.initial_soc = initial_soc
        self.solver = pybamm.IDAKLUSolver()
        self.param_name = param
        self.cell_id = cell_id
        self.create_params()
        self.create_battery(operating_mode=operating_mode)

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
        options =  {"operating mode": operating_mode}
        self.operating_mode = operating_mode
        model = pybamm.lithium_ion.SPM(options=options)
        self.model = model
