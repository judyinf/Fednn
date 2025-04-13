"""
configuration parameters for the models
"""


class LSTMConfig:
    # Data Parameters
    # time_steps = 128

    # Model Parameters
    # input_size = 9  # Number of features in the input data
    # hidden_sizes = [25, 25]  # LSTM layer hidden sizes
    # output_size = 6  # Number of output classes
    input_size = 9  # Number of features in the input data
    hidden_sizes = [100, 100]  # LSTM layer hidden sizes
    output_size = 6  # Number of output classes

    # layer names
    shallow_layer = {'lstm1', 'lstm2'}
    deep_layer = {'fc1', 'fc2'}


class CNNConfig:
    # layer names
    # shallow_key ={'conv1.weight', 'conv1.bias', 'conv2.weight', 'conv2.bias'}
    # deep_key = {'fc1.weight', 'fc1.bias', 'fc2.weight', 'fc2.bias'}
    shallow_layer = {'conv1', 'conv2'}
    deep_layer = {'fc1', 'fc2'}


# configuration parameters for sampling data
class HarConfig:
    labels_list = [0, 1, 2, 3, 4, 5]  # List of unique labels
    Smin = 250  # Minimum number of samples per label
    Smax = 500  # Maximum number of samples per label


class MnistConfig:
    labels_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    Smin = 1000
    Smax = 1600

# class TaskConfig:
#     tensorboard_path = 'tensorboard'
