# Weekly-Challenge-32-Multilayer-Perceptron
For Week 32, I transitioned from traditional Machine Learning into Deep Learning by engineering an **Artificial Neural Network (Multilayer Perceptron)** strictly using NumPy.

To prove the power of hidden layers, I tested the network on a non-linear problem (XOR logic) mapped to an educational scenario. Simple linear models like Logistic Regression cannot mathematically solve XOR boundaries. By introducing a hidden layer and non-linear activation functions, this network can easily discover the complex relationships between the variables.

**How it works**
1. **Forward Propagation:** Data moves from the input nodes, is multiplied by the weight matrices $W$, summed with a bias $b$, and passed through a Sigmoid activation function to the hidden layer, and finally to the output node:
   $$y = \sigma(W \cdot X + b)$$
2. **Backpropagation:** The network calculates the prediction error and uses the Calculus Chain Rule to propagate the error backwards. It computes the partial derivatives (gradients) of the loss function with respect to every single weight in the network.
3. **Gradient Descent:** The model updates its weight matrices in the opposite direction of the gradient to minimize the Mean Squared Error over 10,000 epochs.

**Technical Highlights**
*   **Pure Matrix Calculus:** The backpropagation algorithm is implemented entirely using matrix dot products and vectorized operations in NumPy, avoiding any explicit loops for maximum computational efficiency.

**Dependencies**
* Python 3.14.3
* NumPy
