from FoKL import FoKLRoutines
import pybamm as pb
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# set up FoKL model
model = FoKLRoutines.FoKL(a=9, b=0.01, atau=3, btau=4000, tolerance=4, aic=True, UserWarnings=False)


# create battery model
batmodel = pb.lithium_ion.SPMe()
current = 1

params = pb.ParameterValues("Mohtat2020")
T_max_liion = 80 + 273.15
T_min_liion = -30 + 273.15
T_rands = np.random.rand(1000)
T_inputs = T_rands*(T_max_liion-T_min_liion) + T_min_liion

c_inputs = np.random.rand(1000)

Dn = params["Negative particle diffusivity [m2.s-1]"]

data_struct = Dn(c_inputs,T_inputs)
noise = np.transpose((np.random.rand(1000)-0.5)*1e-1).reshape(-1,1)
data_entries = np.array(data_struct.entries)
dnn = data_entries
dnn_n = np.log(dnn) + noise
dnn_def = np.log(np.sort(data_entries, axis=0))
plt.plot(dnn_def)
plt.show()
inputs = np.transpose(np.array([c_inputs, T_rands]))
struc = np.hstack([inputs, dnn_n])
ii = np.argsort(dnn_n, axis=0)
inputs_sort = inputs[ii]
dnn_sort = dnn_n[ii]
model.fit(inputs_sort,dnn_sort, clean=True)
model.save('DN_model')
model.coverage3(plot=True)