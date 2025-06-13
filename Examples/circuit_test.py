import pybamm
import numpy as np

a = np.array([1,2,3,4,5])
b = np.array([5,4,3,2,1])

c_pb = pybamm.Inner(a,b)
c_np = np.inner(a,b)
print(f"pybamm inner: {c_pb.evaluate()} \n numpy inner: {c_np}")