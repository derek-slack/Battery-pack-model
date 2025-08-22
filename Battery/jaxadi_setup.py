import casadi
import pybamm
import numpy as np
import jaxadi
from jaxadi import convert
import jax
import jax.numpy as jnp

class JaxadiSens:
    def __init__(self):
        self.initialised = False
        self.func = {}
        self.map_init = False

    def sensitivities(self, obj):
        """
        Returns a dictionary of sensitivities for each input parameter.
        The keys are the input parameters, and the value is a matrix of size
        (n_x * n_t, n_p), where n_x is the number of states, n_t is the number of time
        points, and n_p is the size of the input parameter
        """
        # No sensitivities if there are no inputs
        if len(obj.all_inputs[0]) == 0:
            return {}
        # Otherwise initialise and return sensitivities
        if obj._sensitivities is None:
            if obj.all_solution_sensitivities:
                self.initialise_sensitivity_explicit_forward(obj)
            else:
                raise ValueError(
                    "Cannot compute sensitivities. The 'calculate_sensitivities' "
                    "argument of the solver.solve should be changed from 'None' to "
                    "allow sensitivities calculations. Check solver documentation for "
                    "details."
                )
        return self._sensitivities

    def setup_jaxadi(self, casadi_fns):
        for fn in casadi_fns:
            jaxadi_fn = convert(fn, compile=True)
            self.func.update({fn.name(): jax.jit(jax.vmap(jaxadi_fn, in_axes=(0, 0, None)))})
        self.initialised = True


    def initialise_model(self, obj):
        all_S_var = []
        for ts, ys, inputs_stacked, inputs, base_variable, dy_dp in zip(
                obj.all_ts,
                obj.all_ys,
                obj.all_inputs_casadi,
                obj.all_inputs,
                obj.base_variables,
                obj.all_solution_sensitivities['all'],
        ):
            # Set up symbolic variables
            t_casadi = casadi.MX.sym("t")
            y_casadi = casadi.MX.sym("y", ys.shape[0])
            p_casadi = {
                name: casadi.MX.sym(name, value.shape[0])
                for name, value in inputs.items()
            }

            p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])

            # Convert variable to casadi format for differentiating
            var_casadi = base_variable.to_casadi(t_casadi, y_casadi, inputs=p_casadi)
            dvar_dy = casadi.jacobian(var_casadi, y_casadi)
            dvar_dp = casadi.jacobian(var_casadi, p_casadi_stacked)

            # Convert to functions and evaluate index-by-index

            dvar_dy_func = casadi.Function(
                "dvar_dy", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dy]
            )
            dvar_dp_func = casadi.Function(
                "dvar_dp", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dp]
            )

            self.setup_jaxadi([dvar_dy_func, dvar_dp_func])

    def initialise_model_new(self, obj):
        for ts, ys, inputs_stacked, inputs, base_variable, dy_dp in zip(
                obj.all_ts,
                obj.all_ys,
                obj.all_inputs_casadi,
                obj.all_inputs,
                obj.base_variables,
                obj.all_solution_sensitivities['all'],
        ):
            # Set up symbolic CasADi variables
            t_casadi = casadi.MX.sym("t")
            y_casadi = casadi.MX.sym("y", ys.shape[0])
            p_casadi = {
                name: casadi.MX.sym(name, value.shape[0])
                for name, value in inputs.items()
            }
            p_casadi_stacked = casadi.vertcat(*[p for p in p_casadi.values()])

            # Differentiate base variable wrt y and p
            var_casadi = base_variable.to_casadi(t_casadi, y_casadi, inputs=p_casadi)
            dvar_dy = casadi.jacobian(var_casadi, y_casadi)
            dvar_dp = casadi.jacobian(var_casadi, p_casadi_stacked)

            # Build CasADi functions
            dvar_dy_func = casadi.Function("dvar_dy", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dy])
            dvar_dp_func = casadi.Function("dvar_dp", [t_casadi, y_casadi, p_casadi_stacked], [dvar_dp])

            # Convert to JAX without vectorization
            self.func['dvar_dy'] = jaxadi.convert(dvar_dy_func, compile=True)
            self.func['dvar_dp'] = jaxadi.convert(dvar_dp_func, compile=True)

            self.initialised = True

    # def calc_sensitivity(self, all_ts, all_ys, all_inputs, all_solution_sensitivities):
    #     "Set up the sensitivity dictionary"
    #
    #     all_S_var = []
    #     for ts, ys, inputs_stacked, dy_dp in zip(
    #             all_ts,
    #             all_ys,
    #             all_inputs,
    #             all_solution_sensitivities,
    #     ):
    #
    #
    #         ts_jax = jnp.array(ts)  # shape (n_t,)
    #         ys_jax = jnp.array(ys.T)  # shape (n_t, n_y)
    #         p_jax = jnp.array(inputs_stacked)  # shape (n_p,)
    #
    #         # Assuming self.func['dvar_dy'] is jit+vmap wrapped with in_axes=(0, 0, None)
    #         dvar_dy_eval_full = jnp.array(self.func['dvar_dy'](ts_jax, ys_jax, p_jax)[0])
    #
    #         dvar_dy_eval = jnp.block([[dvar_dy_eval_full[0],jnp.zeros(dvar_dy_eval_full[1].shape)],[jnp.zeros(dvar_dy_eval_full[0][0].shape),dvar_dy_eval_full[1][0]]])
    #
    #         dvar_dp_eval = jnp.vstack(
    #
    #                 self.func['dvar_dp'](ts_jax, ys_jax, p_jax)[0]
    #
    #         )
    #
    #         # Compute sensitivity
    #         S_var = dvar_dy_eval @ dy_dp + dvar_dp_eval
    #
    #         all_S_var.append(S_var)
    #
    #     return S_var

    def calc_sensitivity_new(self, all_ts, all_ys, all_inputs, all_solution_sensitivities):
        if not self.map_init:
            self.calc_sens_map_create()

        # Wrap calc_sens_map with jax.jit and mark self as static
        calc_fn = jax.jit(self._calc_sens_map_wrapped, static_argnums=0)
        return calc_fn(self, all_ts, all_ys, all_inputs, all_solution_sensitivities)

    @staticmethod
    def _calc_sens_map_wrapped(self, all_ts, all_ys, all_inputs, all_solution_sensitivities):
        return self.sens_map(all_ts, all_ys, all_inputs, all_solution_sensitivities)
    def calc_sens_map_create(self):
        sens_map_t = jax.vmap(self.calc_sens_i_new, in_axes = (0,0,0,0))
        self.sens_map = sens_map_t
        self.map_init = True

    @jax.jit
    def calc_sens_map(self, all_ts, all_ys, all_inputs, all_solution_sensitivities):
        return self.sens_map(all_ts, all_ys, all_inputs, all_solution_sensitivities)



    def calc_sens_i(self, ts, ys, inputs, dy_dp):


        dvar_dy_eval_full = jnp.array(self.func['dvar_dy'](ts, ys.T, inputs)[0])

        dvar_dy_eval = jnp.block([[dvar_dy_eval_full[0], jnp.zeros(dvar_dy_eval_full[1].shape)],
                                  [jnp.zeros(dvar_dy_eval_full[0][0].shape), dvar_dy_eval_full[1][0]]])

        dvar_dp_eval = jnp.vstack(

            self.func['dvar_dp'](ts, ys.T, inputs)[0]

        )

        # Compute sensitivity
        S_var = jnp.matmul(dvar_dy_eval, dy_dp) + dvar_dp_eval

        return S_var

    def calc_sens_i_new(self, ts, ys, inputs, dy_dp):
        ts_jax = jnp.array(ts)  # shape: (T,)
        ys_jax = jnp.array(ys.T)  # shape: (T, D_y)
        p_jax = jnp.array(inputs)  # shape: (D_p,)
        dy_dp_jax = jnp.array(dy_dp)  # shape: (D_y_total, D_p)

        # Evaluate dvar_dy over time using lax.map
        def eval_one_dy(t_y):
            t, y = t_y
            return self.func['dvar_dy'](t, y, p_jax)  # returns shape (1, D_y)

        dvar_dy_eval_full = jax.lax.map(eval_one_dy, (ts_jax, ys_jax))  # shape: (T, 1, D_y)

        # Collapse leading 1s if needed (for shape matching)
        dvar_dy_eval_flat = jnp.array(dvar_dy_eval_full).reshape(len(ts), -1)  # shape: (T, D_y)

        # Build full Jacobian matrix
        dvar_dy_eval = jnp.kron(jnp.eye(len(ts)), dvar_dy_eval_flat)  # shape: (T*D_out, T*D_y)

        # Evaluate dvar_dp once (if constant over time)
        def eval_one_dp(t_y):
            t, y = t_y
            return self.func['dvar_dp'](t, y, p_jax)  # returns shape (1, D_p)

        dvar_dp_eval = jnp.array(jax.lax.map(eval_one_dp, (ts_jax, ys_jax))) # shape: (T, 1, D_p)
        dvar_dp_eval = dvar_dp_eval.reshape(len(ts) * 1, -1)  # shape: (T, D_p)

        # Compute full S_var: (T*D_out, D_p)
        S_var = dvar_dy_eval @ dy_dp_jax + dvar_dp_eval

        return S_var