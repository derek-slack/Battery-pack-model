import timeit

import numpy as np
import jax
import jax.numpy as jnp
import jax
import jax.numpy as jnp
from jax import jit
from jaxadi import graph_translate as translate
import numpy as np
import scipy
import gc
import psutil
import time

from Battery.jaxadi_setup import JaxadiSens


class Circuit:
    def __init__(self, cells, executor, circuit=None, circuit_from_string = False, use_ids = True):

        self.cells = cells

        if circuit_from_string:
            if type(circuit) is list:
                raise TypeError("When circuit from string is True then entry for circuit should be a string of the circuit value ex: '2s2p' ")
            circuit = self.create_circuit_from_string(circuit)
            self.n_branch = len(circuit)
        else:
            self.n_branch = len(cells)
        if type(circuit) is str:
            raise TypeError("Circuit type is a string, if you want the circuit to be built from a string value then change 'circuit_from_string' to True")

        self.print_circuit(cells)
        self.circuit = circuit
        self.executor = executor
        self.j_sens = [[JaxadiSens(),[]],[[],[]]]
        self.jax_init = False
        self.init_func = None


    def create_circuit_from_string(self, circuit):
        next=False
        num_vec = []
        s = 0
        p = 0
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
        for i in range(p):
            printed_circuit.append([chr(x+k+j) for j in range(s)])
            k += s
            if k >= 26:
                k = 0
        return printed_circuit


    def _equivalent_circuit(self, solution_cells):
        V_branch = np.zeros(len(self.cells))
        I_branch = np.zeros(len(self.cells))
        R_branch = np.zeros(len(self.cells))
        for branch in range(len(self.cells)):
            for cell in range(len(self.cells[branch])):
                V_branch[branch] += solution_cells[branch][cell]["Voltage [V]"]
                # R_branch[branch] += solution_cells[branch][cell]["Resistance [Ohm]"]
                I_branch[branch] = solution_cells[branch][cell]["Current [A]"]

        return V_branch, I_branch#, R_branch

    def _pull_solution(self, solution_t, solution_vars = None, cell_id = False):
        if solution_vars is None:
            solution_vars = ["Voltage [V]", "Current [A]"]
        if cell_id == True:
            solution_vars.append("Cell ID")
        solution_dict = {"Time [s]": solution_t[0][0]["Time [s]"].entries}
        solution_cells = []
        cell_count = 0
        for n_branch in range(len(self.cells)):
            for s_cell in range(len(self.cells[n_branch])):
                solution_cells.append(solution_dict.copy())
                for solution in solution_vars:
                    if solution == "Cell ID":
                        solution_cells[cell_count].update({solution: self.cells[n_branch][s_cell].cell_id})
                    else:
                        solution_cells[cell_count].update({solution: np.array(solution_t[n_branch][s_cell][solution].entries[-1])})
                cell_count += 1
        sol_circuit = self._order_circuit(solution_cells)

        return sol_circuit


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
        for n_branch in range(len(sims)):
            solution_t.append([])
            for i, sim in enumerate(sims[n_branch]):
                test_input = cells[n_branch][i].inputs.copy()

                test_input.update({'Current function [A]': I_new[n_branch]})

                solution_i = solution_tm1[n_branch][i]

                solution_ip1 = sim.step(dt, starting_solution=solution_i, inputs=test_input, calculate_sensitivities=False, t_interp = np.linspace(0,dt,2))
                solution_t[n_branch].append(solution_ip1)

        return solution_t

    def _enter_pool(self, num_workers = 8):
        from concurrent.futures import ThreadPoolExecutor
        executor = ThreadPoolExecutor(max_workers=num_workers)

        return executor

    def _exit_pool(self, executor):
        executor.shutdown()
        pass

    def _unroll_sims(self, sims, cells, solution_tm1, I_new):
        sims_unrolled = []
        cells_unrolled = []
        solution_unrolled = []
        I_new_unrolled = []

        k_vec = []
        k = 0
        for n_branch in range(len(sims)):
            for i in range(len(sims[n_branch])):
                sims_unrolled.append(sims[n_branch][i])
                cells_unrolled.append(cells[n_branch][i])
                solution_unrolled.append(solution_tm1[n_branch][i])
                I_new_unrolled.append(I_new[n_branch])
                k_vec.append([k, n_branch, i])
                k += 1

        return sims_unrolled, cells_unrolled, solution_unrolled, I_new_unrolled, k_vec

    @staticmethod
    def _unroll_for_jaxadi(sol):

        solution_unrolled = []
        k_vec = []
        k = 0
        b=0
        c=0
        for branch in sol:
            for cell in branch:
                solution_unrolled.append(cell)
                k_vec.append([k, b, c])
                k += 1
                c+=1
            b+=1

        return  solution_unrolled, k_vec

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

    def _step_body_p(self, executor, sims, cells, dt, solution_tm1, t, I_new, num_workers=6):

        def _sim_step_par(args):
            sim, cell, dt, sol, input = args
            test_input = cell.inputs.copy()
            test_input.update({'Current function [A]': input})

            sol_np1 = sim.step(dt, starting_solution=sol, inputs=test_input)
            return sol_np1

        sims_ur, cells_ur, sols_ur, I_new_ur, k_vec = self._unroll_sims(sims, cells, solution_tm1, I_new)

        futures = [executor.submit(_sim_step_par, (sim, cell, dt, sol, i_new))
                   for sim, cell, sol, i_new in zip(sims_ur, cells_ur, sols_ur, I_new_ur)]

        sol_p1 = [f.result() for f in futures]

        null, null, sol_rolled = self._roll_sims(sims_ur, cells_ur, sol_p1, k_vec)

        return sol_rolled


    def _order_circuit(self, solution_cells):
        blank_circuit = [[None for _ in range(len(self.cells[branch]))] for branch in range(len(self.cells))]
        k = 0
        for branch in range(len(self.cells)):
            for cell in range(len(self.cells[branch])):
                blank_circuit[branch][cell] = solution_cells[k]
                k+=1
        sol_circuit = blank_circuit
        return sol_circuit


    def solve(self, sims, cells, dt, solution_tm1, t, I_m1, dIdt_m1, old_current_func = None, I_bounds=None):
        # Initial guess from previous solution
        # print(t)
        t1 = timeit.default_timer()
        solution_cells = self._pull_solution(solution_tm1, solution_vars=["Voltage [V]", "Current [A]"])
        t2 = timeit.default_timer() - t1
        print(f"pull time: {t2}")
        V_branch, I_branch = self._equivalent_circuit(solution_cells)
        I0 = np.zeros(len(V_branch))
        if old_current_func is not None:
            I0[0] = old_current_func[0](t)
            I0[1] = old_current_func[1](t)
        else:
            I0 = np.array([(self.P_goal*I_branch[0]/(np.sum(I_branch)))/V_branch[0], (self.P_goal*I_branch[1]/(np.sum(I_branch)))/V_branch[1]])
        # n = len(I0)
        #
        if I_bounds is None:
            I_bounds = [[0, 5],[0,5]]  # Example bounds
        #
        # executor = self.executor


        def simulate_get_voltage(I_test, sens=False):
            """Step simulation forward and pull voltage for candidate current vector."""
            sol_t = self._step_body(sims, cells, dt, solution_tm1, t, I_test)


            sol_dict = self._pull_solution(sol_t, solution_vars=["Voltage [V]", "Current [A]"])

            if sens:
                sens = []
                for n, branch in enumerate(sol_t):
                    sens.append([])
                    for cell in branch:
                        cell['Voltage [V]'].init_func = self.init_func
                        sens_val = cell['Voltage [V]'].sensitivities['Current function [A]']
                        if self.init_func is None:
                            self.init_func = cell['Voltage [V]'].init_func
                        sens[n].append(float(sens_val[-1]))
                sens_sum = np.sum(sens, axis=1)

            V_branch, _ = self._equivalent_circuit(sol_dict)
            if sens:
                return  V_branch, sens_sum, sol_t
            else:
                return V_branch, sol_t

        def total_power(I_test):
            V_branch, _ = simulate_get_voltage(I_test, sens=False)
            # dPdI = 2*((sens_sum*I_test + V_branch)*(V_branch*I_test - self.P_goal))
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
        timer = timeit.default_timer()
        result = scipy.optimize.minimize(objective, I0, bounds=I_bounds, method='SLSQP',  constraints=constraints, jac=False, options={'ftol': 1e-3,'maxiter': 20})
        tttimer = timeit.default_timer() - t1

        print(f"opt time:{tttimer}")
        # result = jax.scipy.optimize.minimize(objective, I0, bounds=I_bounds, method='BFGS',  constraints=constraints, options={'ftol': 1e-5,'maxiter': 20})
        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        # Final simulation with optimal current
        V_branch, sol_new = simulate_get_voltage(result.x)
        P_final = 0
        current_func = []
        dIdt = np.zeros(2)
        dIIdtt = np.zeros(2)
        if I_m1 is None:
            I_m1 = result.x
        if dIdt_m1 is None:
            dIdt_m1 = [0. , 0.]
        for i in range(len(I0)):
            dIdt[i], dIIdtt[i] = self._calculate_derivatives(result.x[i], I_m1[i], dIdt_m1[i], dt)


        function_params = {"Current": result.x, "dCurrent": dIdt, "ddCurrent": dIIdtt, "Time Now": t }
        gc.collect()
        # self._exit_pool(executor)
        P_vec = self._pull_solution(sol_new, solution_vars=["Power [W]"])

        return result.x, P_final, sol_new, function_params, P_vec


    # def solve(self, sims, cells, dt, solution_tm1, t, I_m1, dIdt_m1, old_current_func = None, I_bounds=None):
    #     def pull_sens(sol):
    #         sens = []
    #         for n_branch in range(len(sol)):
    #             sens.append([])
    #             for i in range(len(sol[n_branch])):
    #                 sens[n_branch].append(sol[n_branch][i]['Voltage [V]'].sensitivities[-1])
    #         return sens
    #
    #
    #
    #     dV_dI = pull_sens(solution_tm1)
    #     dV_dI_branch = np.sum(dV_dI, axis=1)
    #     V_i, I_i, R_i = self._equivalent_circuit(solution_tm1)
    #
    #     n_constraints = len(cells)

    @staticmethod
    def _pull_vol_and_sens_i(j_sens, all_ts, all_ys, all_inputs_casadi, all_solution_sensitivities):

        sens = j_sens.calc_sensitivity_new(all_ts, all_ys, all_inputs_casadi, all_solution_sensitivities)

        return sens
    @staticmethod
    def _process_sol_object(obj):
        args = (
                obj.all_ts,
                obj.all_ys,
                obj.all_inputs_casadi,
                obj.all_inputs,
                obj.base_variables,
                obj.all_solution_sensitivities['all'],
        )

        all_ts, all_ys, all_inputs_casadi, all_inputs, base_variables, all_solution_sensitivities = args

        return all_ts, all_ys, all_inputs_casadi, all_inputs, base_variables, all_solution_sensitivities
    def jax_unpack_sens(self, sol):
        sol_ur, k_vec = self._unroll_for_jaxadi(sol)
        all_ts, all_ys, all_inputs_casadi, all_inputs, base_variables, all_solution_sensitivities = [], [], [], [], [], []
        for sol_obj in sol_ur:
            args = self._process_sol_object(sol_obj['Voltage [V]'])
            all_ts.append(args[0])
            all_ys.append(args[1])
            all_inputs_casadi.append(args[2])
            all_inputs.append(args[3])
            base_variables.append(args[4])
            all_solution_sensitivities.append(args[5])
        jnp_all_ts = jnp.array(all_ts)
        jnp_all_ys = jnp.array(all_ys)
        jnp_all_inputs = jnp.array(all_inputs_casadi)
        jnp_all_sens = jnp.array(all_solution_sensitivities)
        # if not self.jax_init:
        #     self.map_vol_and_sens = jax.vmap(self._pull_vol_and_sens_i, in_axes=(None, 0,0,0,0))
        #     self.jax_init = True
        #
        # sens = self.map_vol_and_sens(self.j_sens[0][0], jnp_all_ts, jnp_all_ys, jnp_all_inputs, jnp_all_sens)
        sens = []
        for i in range(4):
            sens.append(self._pull_vol_and_sens_i(self.j_sens[0][0],jnp_all_ts[i], jnp_all_ys[i], jnp_all_inputs[i], jnp_all_sens[i]))
        return sens

    def pull_voltage_and_sensitivities(self, sol):
        V = []
        dVdI = []
        I_cell = []
        branch_n, cell_n = 0,0
        for branch in sol:
            V_branch = 0
            dVdI_branch = 0
            cell_n = 0
            for cell in branch:
                Vt = time.time()
                V_obj = cell['Voltage [V]']
                V_cell = V_obj.entries[-1]
                Vt_end = time.time()
                I_cell.append(cell['Current [A]'].entries[-1])
                It_end = time.time()
                # sens = self.j_sens[0][0].sensitivities(V_obj)['Current function [A]'][-1][0]
                sens = V_obj.sensitivities['Current function [A]'].__getitem__(-1).full()[0,0]
                sens_end = time.time()
                V_branch += V_cell
                dVdI_branch += sens  # Assume wrt current in branch
                adding_end = time.time()
                cell_n += 1
            append_begin = time.time()
            V.append(V_branch)
            dVdI.append(dVdI_branch)
            append_end = time.time()
            gc.collect()
            branch_n += 1

        # print(f"Voltage:{Vt_end- Vt}\n Current:{It_end-Vt_end}\n Sens:{sens_end-It_end}\n adding:{adding_end - sens_end}\n appending:{append_end-append_begin}")
        return np.array(V), np.array(dVdI),  I_cell

    # def solve(self, sims, cells, dt, solution_tm1, t, I_m1, dIdt_m1, old_current_func=None, I_bounds=None):
    #     for k in range(20):
    #         start_time = time.time()
    #         start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    #         t3 = timeit.default_timer()
    #         V, dVdI, I_cell = self.pull_voltage_and_sensitivities(solution_tm1)
    #         if not self.jax_init:
    #             self.j_sens[0][0].initialise_model_new(solution_tm1[0][0]['Voltage [V]'])
    #
    #         start_time = time.time()
    #         # dVdI_ur = self.jax_unpack_sens(solution_tm1)
    #         dVdI_int = []
    #         k = 0
    #         for i in range(2):
    #             dVdI_int.append([])
    #             for j in range(2):
    #                 dVdI_int[i].append(dVdI_ur[k][-1][0][0])
    #                 k+=1
    #
    #         dVdI = np.sum(dVdI_int, axis=1)
    #         end_time = time.time()
    #         end_memory = psutil.Process().memory_info().rss / 1024 / 1024
    #
    #         print(f"sens {k}: {end_time - start_time:.3f}s, "
    #               f"Memory: {start_memory:.1f} -> {end_memory:.1f} MB")
    #
    #         I = np.array([I_cell[0], I_cell[2]])
    #         # Objective: Power penalty
    #         power = np.dot(V, I)
    #         residual = power - self.P_goal
    #         grad_f = 2 * residual * (V + dVdI * I)  # Gradient wrt I
    #         H = 2 * np.outer(V + dVdI * I, V + dVdI * I)  # Gauss-Newton approximation
    #
    #         # Constraints: Voltage matching across branches
    #         c = V[1:] - V[0]  # Shape (b-1,)
    #         J = dVdI.reshape(1,-1) # Shape (b-1, b)
    #
    #         # Build KKT system
    #         KKT_matrix = np.block([
    #             [H, J.T],
    #             [J, np.zeros((J.shape[0], J.shape[0]))]
    #         ])
    #         rhs = -np.concatenate([grad_f, c])
    #         t1 = timeit.default_timer()
    #         delta = np.linalg.solve(KKT_matrix, rhs)
    #         delta_I = delta[:len(I)] * 1e-5
    #         t2 = timeit.default_timer() - t1
    #         print(f"linalg solve {t2}")
    #         # Line search or trust region can be added here
    #         I += delta_I
    #
    #         # Update solution_tm1 with new current
    #         solution_tm1 = self._step_body(sims, cells, dt, solution_tm1, t, I, start=False)  # You’ll need to simulate each branch with new I
    #         gc.collect()
    #     print(residual)
    #     print(c)
    #     dIdt = np.zeros(2)
    #     dIIdtt = np.zeros(2)
    #     if I_m1 is None:
    #         I_m1 = I
    #     if dIdt_m1 is None:
    #         dIdt_m1 = [0. , 0.]
    #     for i in range(len(I)):
    #         dIdt[i], dIIdtt[i] = self._calculate_derivatives(I[i], I_m1[i], dIdt_m1[i], dt)
    #
    #     function_params = {"Current": I, "dCurrent": dIdt, "ddCurrent": dIIdtt, "Time Now": t }
    #
    #     return I, power, solution_tm1, function_params

    def _create_current_poly(self,t_p, I, dIdt, dIIdtt):
        t_p_val = float(t_p)
        I_val = float(I)
        dIdt_val = float(dIdt)
        dIIdtt_val = float(dIIdtt)

        def current_func(t):
            current = I_val + (dIdt_val * (t - t_p_val)) + ((dIIdtt_val * (t - t_p_val) ** 2) / 2)
            return current

        return current_func

    def _calculate_derivatives(self,I, I_m1, dIdt_m1, dt):
        dIdt = (I - I_m1)/dt
        dIIdtt = (dIdt - dIdt_m1)/dt
        return dIdt, dIIdtt
    def print_circuit(self, circuit):

        s = len(circuit[0])
        p = len(circuit)

        print(f"\n{s}s{p}p Circuit: ")
        print("")
        for k in range(len(circuit)):
            print("  ", end='')
            for i, cell in enumerate(circuit[k]):
                if i == 0:
                    n = len(circuit[i])
                    print(f"{cell.cell_id}--", end='')
                elif i > 0 and (i < len(circuit[k]) - 1):
                    print(f"--{cell.cell_id}--", end='')
                else:
                    print(f"--{cell.cell_id}")
            print("  ", end='')
            if k < len(circuit)-1:
                print("|", end='')
                for ii in range(n-1):
                    if ii == 0 or ii == n-1:
                        print("     ", end='')
                    else:
                        print("      ", end='')
                print("|")