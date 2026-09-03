import numpy as np

# Weekly Challenge 32: Multilayer Perceptron
# Author: Ing. Arturo Javier Borbon Rojas

class MultilayerPerceptron:
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate=0.1):

        # Initialize weights with random values and biases with zeros
        self.W1= np.random.uniform(-1,1, (input_nodes, hidden_nodes))
        self.b1= np.zeros((1, hidden_nodes))

        self.W2= np.random.uniform(-1,1, (hidden_nodes, output_nodes))
        self.b2= np.zeros((1, output_nodes))

        self.lr= learning_rate

    def sigmoid(self, x):
        return 1/ (1+np.exp(-x))

    def sigmoid_derivative(self, x):
        # Derivatives of sigmoid requieres the activated output no the raw unput
        return x*(1-x)

    def fit(self,X,y, epochs=10000):
        print(f"Training Neuronal Network over {epochs}- Backpropagation:")

        for epoch in range(epochs):
            # 1 FORWARD PROPAGATION
            hidden_input= np.dot(X,self.W1)+ self.b1
            hidden_output= self.sigmoid(hidden_input)

            final_input= np.dot(hidden_output,self.W2) + self.b2
            final_output= self.sigmoid(final_input)

            # 2. CALCULATE ERROR (Mean Squared Error base)
            error = y - final_output
            
            # 3. BACKWARD PROPAGATION (Calculus Chain Rule)
            # Gradients for output layer
            d_output = error * self.sigmoid_derivative(final_output)
            
            # Gradients for hidden layer
            error_hidden = d_output.dot(self.W2.T)
            d_hidden = error_hidden * self.sigmoid_derivative(hidden_output)

            # 4. UPDATE WEIGHTS AND BIASES (Gradient Descent)
            self.W2 += hidden_output.T.dot(d_output) * self.lr
            self.b2 += np.sum(d_output, axis=0, keepdims=True) * self.lr
            
            self.W1 += X.T.dot(d_hidden) * self.lr
            self.b1 += np.sum(d_hidden, axis=0, keepdims=True) * self.lr

            if epoch % 2000 == 0:
                mse = np.mean(np.square(error))
                print(f"   Epoch {epoch:05d} | Loss (MSE): {mse:.4f}")

    def predict(self, X):
        hidden_output = self.sigmoid(np.dot(X, self.W1) + self.b1)
        final_output = self.sigmoid(np.dot(hidden_output, self.W2) + self.b2)
        return final_output        

            

# --- Non-Linear Educational Dataset (XOR Logic) ---
# Features: [Academic Load (Normalized 0-1), Extracurricular Rest (Normalized 0-1)]
# High Load + Low Rest = 1 (Risk of Burnout)
# Low Load + High Rest = 1 (Risk of Lagging)
# High Load + High Rest = 0 (Balanced/Safe)
# Low Load + Low Rest = 0 (Balanced/Safe)
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]]) # Output labels

# Instantiate Network: 2 Inputs, 4 Hidden Nodes, 1 Output
nn = MultilayerPerceptron(input_nodes=2, hidden_nodes=4, output_nodes=1, learning_rate=0.5)
nn.fit(X_train, y_train, epochs=10000)

print("-" * 65)
print("🔍 EVALUATING NETWORK PREDICTIONS (Non-Linear Boundaries):")

predictions = nn.predict(X_train)
for i, (student, pred) in enumerate(zip(X_train, predictions)):
    predicted_class = 1 if pred[0] > 0.5 else 0
    status = "🚨 At Risk" if predicted_class == 1 else "✅ Balanced (Safe)"
    print(f"Profile [Load: {student[0]}, Rest: {student[1]}] ➔ Risk Probability: {pred[0]*100:05.2f}% | {status}")