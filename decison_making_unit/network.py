import nest
import numpy as np
import pylab
import decison_making_unit.parameters as p
from feature_extraction_unit.lane_extract import *


class DecisionMakingUnit():
    def __init__(self):
        # NEST options
        # This code is a simplified version that the evolved layer was replaced with a simple distance function
        # The evolved layer can be conducted with Braintenberg model proposed in "
        # J. Kaiser et al., Towards a framework for end-to-end control of a simulated vehicle with spiking neural networks,”
        # in 2016 IEEE International Conference on Simulation, Modeling, and Programming for Autonomous Robots (SIMPAR),
        # 2016, pp. 127–134."
        np.set_printoptions(precision=1)
        nest.set_verbosity('M_WARNING')
        nest.ResetKernel()                        
        nest.SetKernelStatus({"local_num_threads": 1, "resolution": p.time_resolution})
        self.featureExtractUnit = FeatureExtractionUnit()
        # Create Poisson neurons
        self.spike_generators = nest.Create("poisson_generator", p.decision_input_size,
                                            params=p.poisson_params)
        self.neuron_pre = nest.Create("parrot_neuron", p.decision_input_size)
        # Create hidden IAF neurons
        self.neuron_hidden = nest.Create("iaf_psc_alpha", p.hidden_num, params=p.iaf_params)
        # Create motor IAF neurons
        self.neuron_post = nest.Create("iaf_psc_alpha", 2, params=p.iaf_params)
        # Create Output spike detector
        self.spike_detector = nest.Create("spike_detector", 2, params={"withgid": True, "withtime": True})
        # Create R2STDP synapses
        self.syn_dict = {"model": "stdp_dopamine_synapse",
                         "weight": {"distribution": "uniform", "low": p.w0_min, "high": p.w0_max}}
        self.syn_dict_hidden = {"model": "stdp_dopamine_synapse",
                         "weight": {"distribution": "uniform", "low": 50.0, "high": 60.0}}
        self.conn_dict = {"rule": "pairwise_bernoulli", 'p': 0.2}
        self.vt = nest.Create("volume_transmitter")
        nest.SetDefaults("stdp_dopamine_synapse",
                         {"vt": self.vt[0], "tau_c": p.tau_c, "tau_n": p.tau_n, "Wmin": p.w_min, "Wmax": p.w_max,
                          "A_plus": p.A_plus, "A_minus": p.A_minus})
        nest.Connect(self.spike_generators, self.neuron_pre, "one_to_one")
        nest.Connect(self.neuron_pre,  self.neuron_hidden, self.conn_dict)
        nest.Connect(self.neuron_hidden, self.neuron_post, "all_to_all", syn_spec=self.syn_dict_hidden)
        nest.Connect(self.neuron_pre, self.neuron_post, "all_to_all", syn_spec=self.syn_dict)
        nest.Connect(self.neuron_post, self.spike_detector, "one_to_one")
        # Create connection handles for left and right motor neuron
        # connection betwenn input layer and output layer
        self.conn_io_l = nest.GetConnections(target=[self.neuron_post[0]])[:p.decision_input_size]
        self.conn_io_r = nest.GetConnections(target=[self.neuron_post[1]])[:p.decision_input_size]
        # connection between hidden layer and output layer
        self.conn_ho_l = nest.GetConnections(target=[self.neuron_post[0]])[p.hidden_num:]
        self.conn_ho_r = nest.GetConnections(target=[self.neuron_post[1]])[p.hidden_num:]
        # connection between input layer and hidden layer
        self.conn_ih_l = nest.GetConnections(target=[self.neuron_hidden[0]])
        self.conn_ih_r = nest.GetConnections(target=[self.neuron_hidden[1]])


    def simulate(self, dvs_data, reward):
        # Set reward signal for left and right network
        nest.SetStatus(self.conn_io_l, {"n": -reward * p.reward_factor})
        nest.SetStatus(self.conn_io_r, {"n": reward * p.reward_factor})
        # Set poisson neuron firing time span
        time = nest.GetKernelStatus("time")
        nest.SetStatus(self.spike_generators, {"origin": time})
        nest.SetStatus(self.spike_generators, {"stop": p.sim_time})
        # Get lane feature from the feature-extraction unit
        feature_potential = self.featureExtractUnit.getLaneFeature(dvs_data)
        # Set poisson neuron firing frequency
        assert feature_potential.size == p.decision_input_size
        feature_potential = feature_potential.reshape(feature_potential.size)
        for i in range(feature_potential.size):
            rate = feature_potential[i] / p.max_potential
            rate = np.clip(rate, 0, 1) * p.max_poisson_freq
            nest.SetStatus([self.spike_generators[i]], {"rate": rate})
        # Simulate network
        nest.Simulate(p.sim_time)
        # Get left and right output spikes
        n_l = nest.GetStatus(self.spike_detector, keys="n_events")[0]
        n_r = nest.GetStatus(self.spike_detector, keys="n_events")[1]
        # Reset output spike detector
        nest.SetStatus(self.spike_detector, {"n_events": 0})
        # Get network weights
        weights_io_l = np.array(nest.GetStatus(self.conn_io_l, keys="weight")).reshape(p.decision_input_size)
        weights_io_r = np.array(nest.GetStatus(self.conn_io_r, keys="weight")).reshape(p.decision_input_size)
        weights_ho_l = np.array(nest.GetStatus(self.conn_ho_l, keys="weight"))
        weights_ho_r = np.array(nest.GetStatus(self.conn_ho_r, keys="weight"))
        weights_ih_l = np.array(nest.GetStatus(self.conn_ih_l, keys="weight"))
        weights_ih_r = np.array(nest.GetStatus(self.conn_ih_r, keys="weight"))
        dSD = nest.GetStatus(self.spike_detector, keys="events")[0]
        return n_l, n_r, weights_io_l, weights_io_r, weights_ho_l, weights_ho_r, weights_ih_l, weights_ih_r, reward, dSD

    def set_weights(self, weights_io_l, weights_io_r, weights_ho_l, weights_ho_r, weights_ih_l, weights_ih_r):
        # Translate weights into dictionary format
        w_io_l = []
        for w in weights_io_l.reshape(weights_io_l.size):
            w_io_l.append({'weight': w})
        w_io_r = []
        for w in weights_io_r.reshape(weights_io_r.size):
            w_io_r.append({'weight': w})
        w_ho_l = []
        for w in weights_ho_l.reshape(weights_ho_l.size):
            w_ho_l.append({'weight': w})
        w_ho_r = []
        for w in weights_ho_r.reshape(weights_ho_r.size):
            w_ho_r.append({'weight': w})
        w_ih_l = []
        for w in weights_ih_l.reshape(weights_ih_l.size):
            w_ih_l.append({'weight': w})
        w_ih_r = []
        for w in weights_ih_r.reshape(weights_ih_r.size):
            w_ih_r.append({'weight': w})
        # Set network weights
        nest.SetStatus(self.conn_io_l, w_io_l)
        nest.SetStatus(self.conn_io_r, w_io_r)
        nest.SetStatus(self.conn_ho_l, w_ho_l)
        nest.SetStatus(self.conn_ho_r, w_ho_r)
        nest.SetStatus(self.conn_ih_l, w_ih_l)
        nest.SetStatus(self.conn_ih_r, w_ih_r)
        return
