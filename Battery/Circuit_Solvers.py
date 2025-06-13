import numpy as np
import scipy.optimize
import jax
import jax.numpy as jnp


class Circuit:
    def __init__(self, cells, circuit, Power_goal, power_tol = 1e-3, vol_tol = 1e-5):

        self.cells = cells
        self.circuit = circuit
        self.print_circuit(circuit)
        self.n_branch = len(circuit)
        self.P_goal = Power_goal
        self.power_tol = power_tol
        # self.P_branch, self.P_diff = self._calculate_branch_power()

    def _equivalent_circuit(self, solution_cells):
        V_branch = np.zeros(len(self.circuit))
        I_branch = np.zeros(len(self.circuit))
        R_branch = np.zeros(len(self.circuit))
        for branch in range(len(self.circuit)):
            for cell in range(len(self.circuit[branch])):
                V_branch[branch] += solution_cells[branch][cell]["Voltage [V]"][-1]
                R_branch[branch] += solution_cells[branch][cell]["Resistance [Ohm]"][-1]
                I_branch[branch] = solution_cells[branch][cell]["Current [A]"][-1]

        return V_branch, I_branch, R_branch

    def _pull_solution(self, solution_t, solution_vars = None):
        if solution_vars is None:
            solution_vars = ["Voltage [V]", "Current [A]"]
        solution_dict = {"Time [s]": solution_t[0]["Time [s]"].entries}
        solution_cells = []
        for i in range(len(self.cells)):
            solution_cells.append(solution_dict.copy())
            for solution in solution_vars:
                solution_cells[i].update({solution: np.array(solution_t[i][solution].entries)})
        sol_circuit = self._order_circuit(solution_cells)

        return sol_circuit
    #
    # def _sol_dict_to_array(self, sol_dict, ):
    #
    #     return sol_array
    def _calculate_new_currents(self):
        P_tot = np.inner(self.V_branch_sort,self.I_branch_sort)
        # P_i_tot = (np.sum(self.I_branch_sort) * V_i)  # Power if every branch had that power
        if P_tot > self.P_goal:
            current_sign = -1 # Current needs to go down
        else:
            current_sign = 1 # Current needs to go up

        # i_close = np.argmin(np.abs(self.P_branch_sort - (self.P_goal / len(self.V_branch_sort))))
        if current_sign == 1:
            i_close = len(self.V_branch_sort) - 1
        elif current_sign == -1:
            i_close = 0

        V_i = self.V_branch_sort[i_close]
        I_i = self.I_branch_sort[i_close]
        R_i = self.R_branch_sort[i_close]

        I_branch_new = np.zeros(self.n_branch)
        for i in range(self.n_branch):
            I_branch_new[i] = abs((V_i - self.V_branch_sort[i])/self.R_branch_sort[i])*current_sign + self.I_branch_sort[i]

        return I_branch_new

    def _sort_branches(self, V_branch, I_branch, R_branch, P_branch, sort_by = "Voltage [V]"):
        if sort_by == "Voltage [V]":
            V_sort = np.sort(V_branch)
            ind_sort = np.argsort(V_branch)
            I_sort = I_branch[ind_sort]
        elif sort_by == "Current [A]":
            I_sort = np.sort(I_branch)
            ind_sort = np.argsort(I_branch)
            V_sort = V_branch[ind_sort]
        else:
            raise TypeError("sort_by sorting must be 'Voltage [V]' or 'Current [A]' ")


        R_sort = R_branch[ind_sort]
        P_sort = P_branch[ind_sort]

        self.V_branch_sort = V_sort
        self.I_branch_sort = I_sort
        self.R_branch_sort = R_sort
        self.P_branch_sort = P_sort

        return V_sort, I_sort, R_sort

    def _calculate_branch_power(self, V_branch, I_branch):
        P_branch = np.zeros(self.n_branch)
        for i in range(self.n_branch):
            P_branch[i] = V_branch[i]*I_branch[i]
        P_diff = np.sum(P_branch) - self.P_goal
        return P_branch, P_diff

    def _vmap_step_func_create_full(self):
        def step_func(dt, sim, solution_i, input_i):
            return sim.step(dt, starting_solution=solution_i, inputs=input_i)

        # First vmap over cells (axis 1 of input_i, axis 0 of sims)
        step_over_cells = jax.vmap(step_func, in_axes=(None, 0, None, 1))

        # Then vmap over input currents (axis 0 of input_i)
        step_over_inputs_and_cells = jax.vmap(step_over_cells, in_axes=(None, None, None, 0))

        return step_over_inputs_and_cells

    def _step_body(self, sims, cells, dt, solution_tm1, t, I_new, start=False):
        solution_t = []
        for i, sim in enumerate(sims):
            test_input = {'Current function [A]': I_new[int(np.floor(i/2))]}

            solution_i = solution_tm1[i]

            solution_ip1 = sim.step(dt, starting_solution=solution_i, inputs=test_input)
            solution_t.append(solution_ip1)

        return solution_t

    def _order_circuit(self, solution_cells):
        blank_circuit = self.circuit.copy()
        k = 0
        for branch in range(len(self.circuit)):
            for cell in range(len(self.circuit)):
                blank_circuit[branch][cell] = solution_cells[k]
                k+=1
        sol_circuit = blank_circuit
        return sol_circuit


    def solve(self, sims, cells, dt, solution_tm1, t, I_bounds=None):
        # Initial guess from previous solution
        print(t)
        solution_cells = self._pull_solution(solution_tm1, solution_vars=["Voltage [V]", "Current [A]", "Resistance [Ohm]"])
        V_branch, I_branch, _ = self._equivalent_circuit(solution_cells)
        I0 = np.array([(self.P_goal*I_branch[0]/(np.sum(I_branch)))/V_branch[0], (self.P_goal*I_branch[1]/(np.sum(I_branch)))/V_branch[1]])
        n = len(I0)

        if I_bounds is None:
            I_bounds = [(-5, 5)] * n  

        def simulate_get_voltage(I_test):
            """Step simulation forward and pull voltage for candidate current vector."""
            sol_t = self._step_body(sims, cells, dt, solution_tm1, t, I_test)
            sol_dict = self._pull_solution(sol_t, solution_vars=["Voltage [V]", "Current [A]", "Resistance [Ohm]"])
            V_branch, _, _ = self._equivalent_circuit(sol_dict)
            return V_branch, sol_t

        def total_power(I_test):
            V_branch, _ = simulate_get_voltage(I_test)
            return np.inner(V_branch, I_test)

        def objective(I_test):
            P = total_power(I_test)
            error = (P - self.P_goal) ** 2

            return error


        def voltage_match_constraint(I_test):
            V_branch, _ = simulate_get_voltage(I_test)
            return V_branch[0] - V_branch[1]

        constraints = [{
            'type': 'eq',
            'fun': voltage_match_constraint
        }]

        result = scipy.optimize.minimize(objective, I0, bounds=I_bounds, method='SLSQP',  constraints=constraints, options={'ftol': 1e-9,'maxiter': 20})

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        # Final simulation with optimal current
        V_branch_final, sol_new = simulate_get_voltage(result.x)
        P_final = np.dot(V_branch_final, result.x)

        return result.x, P_final, sol_new
    # def solve(self, sims, cells, dt, solution_tm1, t, I_bounds=None, V_tol = 1e-5, P_tol = 1e-5):
    #     # Initial guess from previous solution
    #     solution_cells = self._pull_solution(solution_tm1, solution_vars=["Voltage [V]", "Current [A]", "Resistance [Ohm]"])
    #     V_branch, I_branch, _ = self._equivalent_circuit(solution_cells)
    #     I0 = np.array([(self.P_goal*I_branch[0]/(np.sum(I_branch)))/V_branch[0], (self.P_goal*I_branch[1]/(np.sum(I_branch)))/V_branch[1]])
    #     n = len(I0)
    #
    #     if I_bounds is None:
    #         I_bounds = [(-2, 2)] * n
    #
    #     def simulate_get_voltage(I_test):
    #         """Step simulation forward and pull voltage for candidate current vector."""
    #         sol_t = self._step_body(sims, cells, dt, solution_tm1, t, I_test)
    #         sol_dict = self._pull_solution(sol_t, solution_vars=["Voltage [V]", "Current [A]", "Resistance [Ohm]"])
    #         V_branch, _, _ = self._equivalent_circuit(sol_dict)
    #         return V_branch, sol_t
    #
    #     def total_power(I_test):
    #         V_branch, _ = simulate_get_voltage(I_test)
    #         return np.inner(V_branch, I_test)
    #
    #     def objective(I_test):
    #         P = total_power(I_test)
    #         error = (P - self.P_goal) ** 2
    #         return error
    #
    #     def voltage_match_constraint(I_test):
    #         V_branch, _ = simulate_get_voltage(I_test)
    #         return abs(V_branch[0] - V_branch[1]) < V_tol
    #
    #     V_check = False
    #     P_error = objective(np.array([I0, I0, I0, I0]))
    #
    #     while V_check is False or P_error > P_tol:
    #         I_sweep = np.linspace(I_bounds[0], I_bounds[1],25)
    #         V_check
    #
    #     # Final simulation with optimal current
    #     V_branch_final, sol_new = simulate_get_voltage(result.x)
    #     P_final = np.dot(V_branch_final, result.x)
    #
    #     return result.x, P_final, sol_new
    # def solve(self, sims, cells, dt, solution_tm1, t,):
    #
    #     # Initial guess from previous solution
    #     solution_cells = self._pull_solution(solution_tm1, solution_vars=["Voltage [V]", "Current [A]", "Resistance [Ohm]"])
    #     V_branch, I_branch, _ = self._equivalent_circuit(solution_cells)
    #     I0 = np.array([(self.P_goal*I_branch[0]/(np.sum(I_branch)))/V_branch[0], (self.P_goal*I_branch[1]/(np.sum(I_branch)))/V_branch[1]])
    #     n = len(I0)
    #
    #     def simulate_get_voltage_par(I_test):
    #         """Step simulation forward and pull voltage for candidate current vector."""
    #         sol_t = self._step_body(sims, cells, dt, solution_tm1, t, I_test)
    #         sol_dict = self._pull_solution(sol_t, solution_vars=["Voltage [V]", "Current [A]", "Resistance [Ohm]"])
    #         V_branch, _, _ = self._equivalent_circuit(sol_dict)
    #         return V_branch, sol_t
    #
    #     def total_power(I_test):
    #         V_branch, _ = simulate_get_voltage(I_test)
    #         return np.inner(V_branch, I_test)
    #
    #     def objective(I_test):
    #         P = total_power(I_test)
    #         error = (P - self.P_goal) ** 2
    #
    #         return error
    #
    #
    #     def voltage_match_constraint(I_test):
    #         V_branch, _ = simulate_get_voltage(I_test)
    #         return V_branch[0] - V_branch[1]
    #
    #     n_parallel = 20



    # def solve(self, sims, cells, dt, solution_tm1, t):
    #
    #     P_tol = self.power_tol
    #     V_tol = 1e-3
    #     V_diff = 1
    #     tol_i = 15
    #     P_diff = 1.
    #     i = 0
    #     solution_cells = self._pull_solution(solution_tm1, solution_vars=["Voltage [V]", "Current [A]", "Resistance [Ohm]", "Power [W]"])
    #     V_branch, I_branch, R_branch = self._equivalent_circuit(solution_cells)
    #     P_branch, P_diff = self._calculate_branch_power(V_branch, I_branch)
    #     while abs(P_diff) > P_tol and abs(V_diff) > V_tol or i < tol_i:
    #         self._sort_branches(V_branch, I_branch, R_branch, P_branch)
    #         I_new = self._calculate_new_currents()
    #
    #         sol_i = self._step_body(sims, cells, dt, solution_tm1, t, I_new)
    #
    #         sol_dict = self._pull_solution(sol_i,  solution_vars=["Voltage [V]", "Current [A]", "Resistance [Ohm]", "Power [W]"])
    #         V_branch, I_branch, R_branch = self._equivalent_circuit(sol_dict)
    #
    #         P_branch, P_diff = self._calculate_branch_power(V_branch, I_branch)
    #         V_diff = V_branch[1] - V_branch[0]
    #         i += 1
    #     P_tot = np.sum(P_branch)
    #     return I_new, P_tot



    def print_circuit(self, circuit):
        print("\n Circuit:")
        for k in range(len(circuit)):
            for i, cell in enumerate(circuit[k]):
                if i == 0:
                    n = len(circuit[i])
                    print(f"{cell}--", end='')
                elif i > 0 and (i < len(circuit[k]) - 1):
                    print(f"--{cell}--", end='')
                else:
                    print(f"--{cell}")

            if k < len(circuit)-1:
                print("|", end='')
                for ii in range(n-1):
                    if ii == 0 or ii == n-1:
                        print("    ", end='')
                    else:
                        print("     ", end='')
                print("|")
