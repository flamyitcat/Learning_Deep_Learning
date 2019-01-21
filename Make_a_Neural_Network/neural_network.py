import numpy as np
class NeuralNetwork():
    def __init__(self):
        # Seed the random number generator, so it generates the same numbers
        # every time the program runs.
        np.random.seed(1)
        
        # As there are one input layer, hidden layer and output layer
        # it need two weigts matrices
        
        # Assign random weights to a shape of 3 x 4, with range of (1,-1), 
        # as weights between input layer and hidden layer
        self.syn0 = 2 * np.random.random((3,4)) - 1
        
        # Assign random weights to a shape of 4 x 1, with range of (1,-1), 
        # as weights between hidden layer and output layer
        self.syn1 = 2 * np.random.random((4,1)) - 1
        
    def __sigmoid(self, x):
        return 1/(1 + np.exp(-x))
    
    def __sigmoid_derivate(self, x):
        return x*(1-x)

    def train(self, training_set_inputs, training_set_outputs, number_of_training_iteration):
        for iteration in range(number_of_training_iteration):
            
            # Feed forward tjrough layer 0, 1, and 2
            l0 = training_set_inputs
            l1 = self.__sigmoid(np.dot(l0, self.syn0))
            l2 = self.__sigmoid(np.dot(l1, self.syn1))
            
            l2_error = training_set_outputs - l2
            if(iteration % 10000) == 0:
                print("Error:" +str(np.mean(np.abs(l2_error))))

            l2_delta = l2_error*self.__sigmoid_derivate(l2)

            l1_error = l2_delta.dot(self.syn1.transpose())

            l1_delta = l1_error*self.__sigmoid_derivate(l1)

            self.syn1 += l1.transpose().dot(l2_delta)
            self.syn0 += l0.transpose().dot(l1_delta)
    
    def predict(self, inputs):
        return self.__sigmoid(np.dot(self.__sigmoid(inputs.dot(self.syn0)),self.syn1))

if __name__ == "__main__":
    neural_network = NeuralNetwork()

    print("Random starting input_to_hidden_weights:")
    print(neural_network.syn0)
    print("Random starting hidden_to_output_weights:")
    print(neural_network.syn1)

    training_set_inputs = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1]])
    training_set_outputs = np.array([[0, 1, 1, 0]]).transpose()
    
    # train the neural network using a training set
    # Do it 10,000 times and make small adjustments each time
    neural_network.train(training_set_inputs, training_set_outputs, 60000)

    print("New input_to_hidden_weights after training:")
    print(neural_network.syn0)
    print("New hidden_to_output_weights after training:")
    print(neural_network.syn1)

    print("Considering new situation [1,0,0] -> ??? :")
    print(neural_network.predict(np.array([1,0,0])))
