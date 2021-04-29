import numpy as np

class Network():

    def __init__(self, codebook):

        self.codebook = codebook
        self.W1 = np.random.normal(0, 1, size = (16,len(codebook)))
        self.W21 = np.random.normal(0, 1, size = (128, 16))
        self.W22 = np.random.normal(0, 1, size = (128, 16))
        self.W23= np.random.normal(0, 1, size =(128, 16))
        self.W3 = np.random.normal(0, 1, size =(len(codebook), 128))
        self.bias1 = np.random.normal(0, 1, size= (128, 1))
        self.bias2 = np.random.normal(0, 1, size=(len(codebook),1))


    def sigmoid(self, S):

        return (1 / (1 + np.exp(-S)))

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    # def softmax(self, S):
        
    #     return np.exp(S) / np.exp(S).sum()

    def stable_softmax(self, S):
        z = S - np.max(S)
        result = np.exp(z) / np.sum(np.exp(z))
        return result


    def forward_propogation(self, Word1_encoding, Word2_encoding, Word3_encoding):

        embedding_layer1 = self.W1 @ Word1_encoding.T
        embedding_layer2 = self.W1 @ Word2_encoding.T 
        embedding_layer3 = self.W1 @ Word3_encoding.T

        hidden_layer_1 = self.W21 @ embedding_layer1
        hidden_layer_2 = self.W22 @ embedding_layer2
        hidden_layer_3 = self.W23 @ embedding_layer3

        hidden_layer = self.sigmoid(hidden_layer_1 + hidden_layer_2 + hidden_layer_3 + self.bias1)

        prediction = self.stable_softmax(self.W3 @ hidden_layer + self.bias2)

        return embedding_layer1, embedding_layer2, embedding_layer3, hidden_layer, hidden_layer_1, hidden_layer_2, hidden_layer_3, prediction.T


    def cross_entropy_loss(self, Y, prediction):
	    return -np.sum(Y * np.log(prediction)) * 1/len(Y)


    def backpropagation(self,Word1_encoding, Word2_encoding, Word3_encoding, target):
        
        embedding_layer1, embedding_layer2, embedding_layer3, hidden_layer, hidden_layer_1, hidden_layer_2, hidden_layer_3, prediction = self.forward_propogation(Word1_encoding, Word2_encoding, Word3_encoding) #bunu ayır

        loss = self.cross_entropy_loss(target, prediction)

        derivative_of_prediction_wrt_softmax  = (prediction - target) * 1/len(target)

        derivative_of_loss_wrt_bias2 =  np.sum(derivative_of_prediction_wrt_softmax.T, axis=1, keepdims=True)

        derivative_of_loss_wrt_W3 = (hidden_layer @ derivative_of_prediction_wrt_softmax).T

        derivative_of_loss_wrt_hidden = (derivative_of_prediction_wrt_softmax @ self.W3).T

        sigmoid_derivative = self.sigmoid_derivative(hidden_layer_1 + hidden_layer_2 + hidden_layer_3 + self.bias1)
        
        derivative_of_loss_wrt_sigmoid = derivative_of_loss_wrt_hidden * sigmoid_derivative

        derivative_of_loss_wrt_bias1 = np.sum(derivative_of_loss_wrt_sigmoid, axis=1, keepdims=True)

        derivative_of_loss_wrt_W21 = derivative_of_loss_wrt_sigmoid @ embedding_layer1.T
        derivative_of_loss_wrt_W22 = derivative_of_loss_wrt_sigmoid @ embedding_layer2.T
        derivative_of_loss_wrt_W23 = derivative_of_loss_wrt_sigmoid @ embedding_layer3.T

        derivative_of_loss_wrt_embedding_1 = self.W21.T @ derivative_of_loss_wrt_sigmoid
        derivative_of_loss_wrt_embedding_2 = self.W22.T @ derivative_of_loss_wrt_sigmoid
        derivative_of_loss_wrt_embedding_3 = self.W23.T @ derivative_of_loss_wrt_sigmoid

        derivative_of_loss_wrt_W1 = derivative_of_loss_wrt_embedding_1 @ Word1_encoding + derivative_of_loss_wrt_embedding_2 @ Word2_encoding + derivative_of_loss_wrt_embedding_3 @ Word3_encoding

        return derivative_of_loss_wrt_W1, derivative_of_loss_wrt_W21, derivative_of_loss_wrt_W22, derivative_of_loss_wrt_W23, derivative_of_loss_wrt_W3, derivative_of_loss_wrt_bias1, derivative_of_loss_wrt_bias2, loss


