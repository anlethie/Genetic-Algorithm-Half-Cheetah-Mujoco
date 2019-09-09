from functools import reduce
import numpy as np


def product(l):
    return reduce(lambda x,y: x*y, l, 1)

def space_size(space):
    try:
        return space.n
    except AttributeError:
        pass
    try:
        return product(space_size(s) for s in space.spaces)
    except AttributeError:
        pass
    try:
        return product(space.shape)
    except AttributeError:
        print('Could not interpret size of space', space)
        return None

def normalize_observation(obs, obs_space):
    def asfloat(x):
        return np.array(x, dtype=float)
    try:
        return (
            (asfloat(obs) - asfloat(obs_space.low)) /
            (asfloat(obs_space.high) - asfloat(obs_space.low))
            )
    except AttributeError:
        pass
    try:
        output = np.zeros(obs_space.n)
        output[obs] = 1. # one-hot interpretation of a discrete space
        return output
    except AttributeError:
        pass
    try:
        output = np.array([])
        for o,s in zip(obs,obs_space.spaces):
            output = np.concatenate((output, normalize_observation(o, s)))
        return np.reshape(output, product(output.shape)).copy()
    except AttributeError:
        print('Could not recognize observation', obs, 'and space', obs_space)
        return None

def interpret_action(num, action_space):
    """Interprets discrete action num as an action in action_space"""
    try:
        output = []
        for s in action_space.spaces:
            output.append(num % s.n)
            num = num // s.n
        return output
    except AttributeError:
        pass
    return num

class Actor:
    """Encapsulates how to respond to observations in some environment."""
    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space      = action_space
        self._n_obs = space_size(observation_space)
        self._n_act = space_size(action_space)

    def react_to(self, observation):
        """Returns an action in response to the observation."""
        # Without specializing, this is just a random actor
        return self._action_space.sample()


class GeneticActor(Actor):
    def get_genome(self):
        """Returns the genomic representation of self. A genome is a numpy array of floats."""
        return np.array([])

    def from_genome(self, genome):
        """Generates a new GeneticActor for the same environment as this actor, based on the given genome."""
        return GeneticActor(self._observation_space, self._action_space)



class PerceptronActor(Actor):
    def __init__(self, observation_space, action_space):
        super(PerceptronActor, self).__init__(observation_space, action_space)
        self._perceptron_matrix = (np.random.random((self._n_act, self._n_obs))*2)-1

    def react_to(self, observation):
        obs = normalize_observation(observation, self._observation_space)
        outputs = self._perceptron_matrix.dot(obs)
        i = 0
        for j,x in enumerate(outputs):
            if x > outputs[i]:
                i = j
        return interpret_action(i, self._action_space)

class NeuralNetActor(Actor):
    """Actor that uses a neural network to react to observations."""
    def __init__(self, observation_space, action_space, hidden_layers=[]):
        """hidden_layers is a list of numbers, each number the number of nodes on a hidden layer, in order"""
        super(NeuralNetActor, self).__init__(observation_space, action_space)
        self._layers = []
        for in_size,out_size in zip([self._n_obs] + hidden_layers, hidden_layers + [self._n_act]):
            self._layers.append((np.random.random((out_size, in_size))*2)-1)
        # self._threshold_fn = lambda X: 1. / (1. + np.exp(-X))

    def react_to(self, observation):
        obs = normalize_observation(observation, self._observation_space)
        current_vector = np.reshape(obs, self._n_obs)
        for layer in self._layers:
            current_vector = layer.dot(current_vector)
            # current_vector = self._threshold_fn(current_vector)
            current_vector = 1. / (1. + np.exp(-current_vector))
        i = 0
        for j,x in enumerate(current_vector):
            if x > current_vector[i]:
                i = j
        return interpret_action(i, self._action_space)

class GeneticPerceptronActor(PerceptronActor, GeneticActor):
    def get_genome(self):
        return (np.reshape(self._perceptron_matrix.copy(), self._n_obs * self._n_act) + 1) / 2

    def from_genome(self, genome):
        pa = GeneticPerceptronActor(self._observation_space, self._action_space)
        pa._perceptron_matrix = (2 * np.reshape(genome.copy(), self._perceptron_matrix.shape)) - 1
        return pa

class GeneticNNActor(NeuralNetActor, GeneticActor):
    def get_genome(self):
        genome = np.array([])
        for layer in self._layers:
            genome = np.concatenate((genome, np.reshape(layer.copy(), product(layer.shape))))
        genome = (genome + 1.)/2.
        return genome

    def from_genome(self, genome):
        genome = (genome.copy() * 2.) - 1.
        nna = GeneticNNActor(self._observation_space, self._action_space)
        nna._layers = []
        start = 0
        for layer in self._layers:
            layer_size = product(layer.shape)
            new_layer = np.reshape(genome[start:start+layer_size].copy(), layer.shape)
            nna._layers.append(new_layer)
            start += layer_size
        return nna


class ModifiedNeuralNetActor(Actor):
    """Actor that uses a neural network to react to observations."""
    def __init__(self, observation_space, action_space, hidden_layers=[]):
        """hidden_layers is a list of numbers, each number the number of nodes on a hidden layer, in order"""
        super(ModifiedNeuralNetActor, self).__init__(observation_space, action_space)
        self._layers = []
        self._layers_bias = []
        self._n_act = action_space.low.shape[0]
        self._n_obs = product(observation_space.shape)

        # 1 neuron has multiple weights
        for in_size,out_size in zip([self._n_obs] + hidden_layers, hidden_layers + [self._n_act]):
            self._layers.append((np.random.uniform(low=-1, high=1, size=(out_size, in_size))))

        # 1 neuron has 1 bias
        for num_neuron in hidden_layers:    
            self._layers_bias.append((np.random.uniform(low=-1, high=1, size=(num_neuron))))
        # Also add bias for the output layer
        self._layers_bias.append((np.random.uniform(low=-1, high=1, size=(self._n_act))))
        # self._threshold_fn = lambda X: 1. / (1. + np.exp(-X))

    def react_to(self, observation):
        # Since mujoco observation space is [inf, inf], input normalization is not required
        # obs = normalize_observation(observation, self._observation_space)
        current_vector = np.reshape(observation, self._n_obs)

        # Foward feed through all hidden layers, except output layer
        for i in range(len(self._layers)-1):
            current_vector = self._layers[i].dot(current_vector) + self._layers_bias[i]
            #current_vector = np.tanh(current_vector) # Tanh activation
            current_vector = current_vector * (current_vector > 0) # Relu activation
        
        # unbounded output vector [-inf, inf]
        current_vector = self._layers[-1].dot(current_vector) + self._layers_bias[-1]

        # normalize to [-1,1]
        #current_vector = np.tanh(current_vector)

        return current_vector


class ModifiedGeneticNNActor(ModifiedNeuralNetActor, GeneticActor):
    def get_genome(self):
        genome = np.array([])
        genome_bias = np.array([])
        for layer in self._layers:
            genome = np.concatenate((genome, np.reshape(layer.copy(), product(layer.shape))))
        # print("Genome's shape: " + str(np.asarray(self._layers).shape))
        # print(self._layers[1].shape)
        for layer_bias in self._layers_bias:
            genome_bias = np.concatenate((genome_bias, np.reshape(layer_bias.copy(), product(layer_bias.shape))))
        # print("Bias's shape: " + str(np.asarray(self._layers_bias).shape))
        # print(self._layers_bias[1].shape)

        genome = np.concatenate((genome, genome_bias))
        # print("Total genome's shape: " + str(genome.shape) )
        
        genome = (genome + 1.)/2.

        # print("===============================================")

        return genome

    def from_genome(self, genome):
        genome = (genome.copy() * 2.) - 1.
        nna = ModifiedGeneticNNActor(self._observation_space, self._action_space)
        nna._layers = []
        nna._layers_bias = []
        start = 0
        for layer in self._layers:
            layer_size = product(layer.shape)
            new_layer = np.reshape(genome[start:start+layer_size].copy(), layer.shape)
            nna._layers.append(new_layer)
            start += layer_size
        for layer in self._layers_bias:
            layer_size = product(layer.shape)
            new_layer = np.reshape(genome[start:start+layer_size].copy(), layer.shape)
            nna._layers_bias.append(new_layer)
            start += layer_size
        return nna