import numpy as np

class Network():

    def __init__(self, codebook):

        self.codebook = codebook
        self.W1 = np.random.uniform(low=0, high=0.05, size = (16,len(codebook)))
        self.W21 = np.random.uniform(low=0, high=0.05, size = (128, 16))
        self.W22 = np.random.uniform(low=0, high=0.05, size = (128, 16))
        self.W23= np.random.uniform(low=0, high=0.05, size =(128, 16))
        self.W3 = np.random.uniform(low=0, high=0.05, size =(len(codebook), 128))
        self.bias1 = np.random.uniform(low=0, high=0.05, size= (128,1))
        self.bias2 = np.random.uniform(low=0, high=0.05, size=(len(codebook),1))


    def sigmoid(self, S):

        return (1 / (1 + np.exp(-S)))

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    # def softmax(self, S):
        
    #     return np.exp(S) / np.sum(np.exp(S))

    def softmax(self, Z):
        expZ = np.exp(Z - np.max(Z))
        return expZ / expZ.sum(axis=0, keepdims=True)


    def forward_propogation(self, Word1_encoding, Word2_encoding, Word3_encoding):

        embedding_layer1 = self.W1 @ Word1_encoding.T
        embedding_layer2 = self.W1 @ Word2_encoding.T 
        embedding_layer3 = self.W1 @ Word3_encoding.T

        hidden_layer_1 = self.sigmoid(self.W21 @ embedding_layer1  + self.bias1)
        hidden_layer_2 = self.sigmoid(self.W22 @ embedding_layer2  + self.bias1)
        hidden_layer_3 = self.sigmoid(self.W23 @ embedding_layer3  + self.bias1)
        hidden_layer = hidden_layer_1 + hidden_layer_2 + hidden_layer_3
        
        prediction = self.softmax(self.W3 @ hidden_layer + self.bias2)

        return embedding_layer1, embedding_layer2, embedding_layer3, hidden_layer, prediction.T


    def cross_entropy_loss(self, Y, prediction):
	    return -np.mean(Y * np.log(prediction))

    def backpropagation(self,Word1_encoding, Word2_encoding, Word3_encoding, target):

        embedding_layer1, embedding_layer2, embedding_layer3, hidden_layer, prediction = self.forward_propogation(Word1_encoding, Word2_encoding, Word3_encoding)

        loss = self.cross_entropy_loss(target, prediction)

        derivative_of_prediction_wrt_softmax  = (prediction - target) * 1/len(target)

        derivative_of_softmax_wrt_bias2 = np.ones([len(target),1])

        derivative_of_softmax_wrt_bias1 = np.ones([len(target),1])

        derivative_of_softmax_wrt_W3 = hidden_layer


        derivative_of_loss_wrt_W3 = (derivative_of_softmax_wrt_W3 @ derivative_of_prediction_wrt_softmax).T
        derivative_of_loss_wrt_bias2 =  derivative_of_prediction_wrt_softmax.T @ derivative_of_softmax_wrt_bias2

        derivative_of_loss_wrt_W21 = (derivative_of_prediction_wrt_softmax @ (self.sigmoid_derivative(derivative_of_loss_wrt_W3)*(1-self.sigmoid_derivative(derivative_of_loss_wrt_W3)))).T @ embedding_layer1.T
        derivative_of_loss_wrt_W22 = (derivative_of_prediction_wrt_softmax @ (self.sigmoid_derivative(derivative_of_loss_wrt_W3)*(1-self.sigmoid_derivative(derivative_of_loss_wrt_W3)))).T @ embedding_layer2.T
        derivative_of_loss_wrt_W23 = (derivative_of_prediction_wrt_softmax @ (self.sigmoid_derivative(derivative_of_loss_wrt_W3)*(1-self.sigmoid_derivative(derivative_of_loss_wrt_W3)))).T @ embedding_layer3.T

        derivative_of_loss_wrt_W1 = (derivative_of_prediction_wrt_softmax @ (self.sigmoid_derivative(derivative_of_loss_wrt_W3)*(1-self.sigmoid_derivative(derivative_of_loss_wrt_W3))) @ self.W21).T @ Word1_encoding\
            + (derivative_of_prediction_wrt_softmax @ (self.sigmoid_derivative(derivative_of_loss_wrt_W3)*(1-self.sigmoid_derivative(derivative_of_loss_wrt_W3))) @ self.W22).T @ Word2_encoding\
            + (derivative_of_prediction_wrt_softmax @ (self.sigmoid_derivative(derivative_of_loss_wrt_W3)*(1-self.sigmoid_derivative(derivative_of_loss_wrt_W3))) @ self.W23).T @ Word3_encoding

        derivative_of_loss_wrt_bias1 =  (derivative_of_prediction_wrt_softmax @ (self.sigmoid_derivative(derivative_of_loss_wrt_W3)*(1-self.sigmoid_derivative(derivative_of_loss_wrt_W3)))).T @ derivative_of_softmax_wrt_bias1\
            + (derivative_of_prediction_wrt_softmax @ (self.sigmoid_derivative(derivative_of_loss_wrt_W3)*(1-self.sigmoid_derivative(derivative_of_loss_wrt_W3)))).T @ derivative_of_softmax_wrt_bias1\
            + (derivative_of_prediction_wrt_softmax @ (self.sigmoid_derivative(derivative_of_loss_wrt_W3)*(1-self.sigmoid_derivative(derivative_of_loss_wrt_W3)))).T @ derivative_of_softmax_wrt_bias1

        

        return derivative_of_loss_wrt_W1, derivative_of_loss_wrt_W21, derivative_of_loss_wrt_W22, derivative_of_loss_wrt_W23, derivative_of_loss_wrt_W3, derivative_of_loss_wrt_bias1, derivative_of_loss_wrt_bias2, loss






