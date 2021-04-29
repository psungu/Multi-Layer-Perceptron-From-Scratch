import numpy as np

class Network():

    def __init__(self, codebook):

        self.codebook = codebook
        self.W1 = np.random.normal(0, 0.01, size = (16,len(codebook)))
        self.W2 = np.random.normal(0, 0.01, size = (128, 48))
        self.W3 = np.random.normal(0, 0.01, size =(len(codebook), 128))
        self.bias1 = np.zeros([128, 1])
        self.bias2 = np.zeros([len(codebook),1])


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

        embedding_layer = np.concatenate([embedding_layer1, embedding_layer2, embedding_layer3], axis=0)

        hidden_input = self.W2 @ embedding_layer + self.bias1

        hidden_layer = self.sigmoid(hidden_input)

        prediction = self.stable_softmax(self.W3 @ hidden_layer + self.bias2)

        return embedding_layer, hidden_layer, hidden_input, prediction.T


    def cross_entropy_loss(self, Y, prediction):
        
	    return -np.sum(Y * np.log(prediction)) * 1/len(Y)


    def backpropagation(self,Word1_encoding, Word2_encoding, Word3_encoding, target):
        
        embedding, hidden_layer, hidden_input, prediction = self.forward_propogation(Word1_encoding, Word2_encoding, Word3_encoding) 

        loss = self.cross_entropy_loss(target, prediction)

        derivative_of_loss_wrt_softmax  = (prediction - target) * 1/len(target)

        derivative_of_loss_wrt_bias2 =  np.sum(derivative_of_loss_wrt_softmax.T, axis=1, keepdims=True)

        derivative_of_loss_wrt_W3 = (hidden_layer @ derivative_of_loss_wrt_softmax).T

        derivative_of_loss_wrt_hidden = (derivative_of_loss_wrt_softmax @ self.W3).T

        sigmoid_derivative = self.sigmoid_derivative(hidden_input)
        
        derivative_of_loss_wrt_sigmoid = derivative_of_loss_wrt_hidden * sigmoid_derivative

        derivative_of_loss_wrt_bias1 = np.sum(derivative_of_loss_wrt_sigmoid, axis=1, keepdims=True)

        derivative_of_loss_wrt_W2 = derivative_of_loss_wrt_sigmoid @ embedding.T

        derivative_of_loss_wrt_embedding = self.W2.T @ derivative_of_loss_wrt_sigmoid

        derivative_of_loss_wrt_W1 = derivative_of_loss_wrt_embedding[0:16:] @ Word1_encoding + derivative_of_loss_wrt_embedding[16:32:] @ Word2_encoding + derivative_of_loss_wrt_embedding[32:] @ Word3_encoding

        return derivative_of_loss_wrt_W1, derivative_of_loss_wrt_W2, derivative_of_loss_wrt_W3, derivative_of_loss_wrt_bias1, derivative_of_loss_wrt_bias2, loss




