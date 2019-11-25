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
weights_io_l = []
weights_io_r = []
weights_ho_l = []
weights_ho_r = []
weights_ih_l = []
weights_ih_r = []
rewards = []
episode_position_o = []
episode_i_o = []
episode_position_i = []
episode_i_i = []
coordinate_list = []
distance_list = []

# Initialize environment, get initial state, initial reward
s, r = env.reset()
# cmap = cm.Blues
# fig = plt.figure()

# fig_weights, ax_weights = plt.subplots()
evs = []
ts = []
for i in range(training_length):
    # Simulate network for 50 ms
    # get number of output spikes and network weights
    n_l, n_r, w_io_l, w_io_r, w_ho_l, w_ho_r, w_ih_l, w_ih_r, reward, dSD = snn.simulate(s, r)
    print("nl：", n_l, "; nr： ", n_r)
    s, d, p, r, t, n, o = env.step(n_l, n_r)
    coordinate_list.append(env.get_coordinate())
    distance_list.append(p)

    # Save weights every simulation steps, you can set
    print("================>step", i, " / ", training_length, "<==================")
    print("position: ", p)
    print("reward: ", reward)
    rewards.append(reward)
    weights_io_l.append(w_io_l)
    weights_io_r.append(w_io_r)
    weights_ho_l.append(w_ho_l)
    weights_ho_r.append(w_ho_r)
    weights_ih_l.append(w_ih_l)
    weights_ih_r.append(w_ih_r)
    # weight_selected_sequence1.append(w_io_l[4][3])
    # weight_selected_sequence2.append(w_io_l[8][7])
    # weight_selected_sequence3.append(w_io_l[11][3])
    if i == training_length-1:
        curr_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        pylab.figure(3)
        print(ts)
        pylab.plot(ts, evs, ".")
        pylab.show()
        np.save("conditional_learning_weights_io_l" + str(datetime.date.today()) + ".npy", weights_io_l)
        np.save("conditional_learning_weights_io_r" + str(datetime.date.today()) + ".npy", weights_io_r)
        np.save("conditional_learning_weights_ho_l" + str(datetime.date.today()) + ".npy", weights_ho_l)
        np.save("conditional_learning_weights_ho_r" + str(datetime.date.today()) + ".npy", weights_ho_r)
        np.save("conditional_learning_weights_ih_l" + str(datetime.date.today()) + ".npy", weights_ih_l)
        np.save("conditional_learning_weights_ih_r" + str(datetime.date.today()) + ".npy", weights_ih_r)
        np.save("conditional_learning_coordinate_list_" + str(curr_time) + ".npy", coordinate_list)
        np.save("conditional_learning_distance_list_" + str(curr_time) + ".npy", distance_list)
        plt.figure()
        cmap = cm.Blues
        w_l_transe = np.array(w_io_l).transpose()
        ax_left = plt.subplot(211)
        map_left = ax_left.imshow(w_l_transe, cmap=cmap)
        colorbar_left = plt.colorbar(mappable=map_left)
        ax_left.set_title("weights to left motor neuron")

        w_r_transe = np.array(w_io_r).transpose()
        ax_right = plt.subplot(212)

        ax_right.clear()
        map_right = ax_right.imshow(w_r_transe, cmap=cmap)
        colorbar_right = plt.colorbar(mappable=map_right)
        ax_right.set_title("weights to right motor neuron")
        plt.tight_layout()

        plt.figure(2)
        plt.plot(rewards)
        np.save("rewards" + str(datetime.date.today()) + ".npy", rewards)
        plt.title("Variation of Reward Value")
        plt.figure(3)
        positions.pop()
        plt.plot(positions)
        np.save("positons" + str(datetime.date.today()) + ".npy", positions)
        plt.title("Variation of Moving Distance")
        plt.figure(4)
        plt.plot(weight_selected_sequence1, label='Weight Value of Neuron1')
        plt.plot(weight_selected_sequence2, label='Weight Value of Neuron2')
        plt.plot(weight_selected_sequence3, label='Weight Value of Neuron3')
        plt.legend()
        plt.xlabel("Episode")
        plt.ylabel("Weight Value")
        plt.show()




