import matplotlib.pyplot as plt
import nnfs
from nnfs.datasets import vertical_data

nnfs.init()

# Create dataset
X, y = vertical_data(samples=100, classes=3)

# Create model
dense1 = Layer_Dense(2, 3) # first dense layer, 2 inputs
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3) # second dense layer, 3 inputs, 3 outputs

# Create loss function
loss_function = Loss_CategoricalCrossentropy()

# Helper variables
lowest_loss = 9999999 # some initial value
best_dense1_weights = 0.05 * np.random.randn(2, 3)
best_dense1_biases = 0.05 * np.random.randn(1, 3)
best_dense2_weights = 0.05 * np.random.randn(3, 3)
best_dense2_biases = 0.05 * np.random.randn(1, 3)
