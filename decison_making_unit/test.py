from matplotlib import cm

from decison_making_unit.network import *
from decison_making_unit.environment_1 import *
from decison_making_unit.parameters import *
import matplotlib.pyplot as plt
# import h5py
from pyheatmap.heatmap import HeatMap
import numpy as np
import datetime
import pylab

snn = DecisionMakingUnit()
env = VrepEnvironment()
coordinate_list = []


# Initialize environment, get initial state, initial reward
s, r = env.reset()
weights_io_l = np.load("weights_io_l2019-04-17.npy")
weights_io_r = np.load("weights_io_r2019-04-17.npy")
weights_ho_l = np.load("weights_ho_l2019-04-17.npy")
weights_ho_r = np.load("weights_ho_r2019-04-17.npy")
weights_ih_l = np.load("weights_ih_l2019-04-17.npy")
weights_ih_r = np.load("weights_ih_r2019-04-17.npy")
snn.set_weights(weights_io_l, weights_io_r, weights_ho_l, weights_ho_r, weights_ih_l, weights_ih_r)

# fig_weights, ax_weights = plt.subplots()
evs = []
ts = []

for i in range(1000):
    # Simulate network for 50 ms
    # get number of output spikes and network weights
    n_l, n_r, w_io_l, w_io_r, w_ho_l, w_ho_r, w_ih_l, w_ih_r, reward, dSD = snn.simulate(s, r)
    print("nl：", n_l, "; nr： ", n_r)


    s, d, p, r, t, n, o = env.step(n_l, n_r)
    coordinate_list.append(env.get_coordinate())

    print("================>step", i, " / ", training_length, "<==================")

    if i == training_length-1:
        curr_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        pylab.figure(3)
        print(ts)
        pylab.plot(ts, evs, ".")
        pylab.show()
        np.save("test_coordinate_list_" + str(curr_time) + ".npy", coordinate_list)





