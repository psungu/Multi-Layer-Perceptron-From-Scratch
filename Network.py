import numpy as np


X = np.load('train_inputs.npy')[:100]
Y = np.load('train_targets.npy')[:100]
codebook =  np.load('vocab.npy')

word4_target = np.eye(len(codebook))[Y]

word1_target = X[:,0].reshape(-1)
Word1_encoding = np.eye(len(codebook))[word1_target]

word2_target = X[:,1].reshape(-1)
Word2_encoding = np.eye(len(codebook))[word2_target]

word3_target = X[:,2].reshape(-1)
Word3_encoding = np.eye(len(codebook))[word3_target]


class Network():

    def __init__(self, train_data, codebook):
        np.random.seed(1)
        self.X = train_data
        self.codebook = codebook
        self.W1 = np.random.rand(16,len(codebook))
        self.W21 = np.random.rand(128, 16)
        self.W22 = np.random.rand(128, 16)
        self.W23= np.random.rand(128, 16)
        self.W3 = np.random.rand(len(codebook), 128)
        self.W3_2 = np.random.rand(len(codebook), 128)
        self.W3_3 = np.random.rand(len(codebook), 128)
        self.bias1 = np.random.rand(128,1)

        self.bias2= np.random.rand(len(codebook),1)

        # self.W1_initialization = np.ones([16,len(codebook)])
        # self.W2_1_initialization = np.ones([128, 16])
        # self.W2_2_initialization = np.ones([128, 16])
        # self.W2_3_initialization = np.ones([128, 16])
        # self.W3_1_initialization = np.ones([len(codebook), 128])
        # self.W3_2_initialization = np.ones([len(codebook), 128])
        # self.W3_3_initialization = np.ones([len(codebook), 128])
        # self.bias1_1_initialization = np.ones([128,1])
        # self.bias1_2_initialization = np.ones([128,1])
        # self.bias1_3_initialization = np.ones([128,1])

        # self.bias2_initialization = np.ones([len(codebook),1])

    def sigmoid(self, S):

        return (1 / (1 + np.exp(-S)))

    def softmax(self, S):
        
        return np.exp(S) / np.sum(np.exp(S), axis=0)

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
	    return -np.sum(Y.T @ np.log(prediction)) * 1/len(Y)



    def backward_propagation(self,Word1_encoding, Word2_encoding, Word3_encoding, target):

        embedding_layer1, embedding_layer2, embedding_layer3, hidden_layer, prediction = self.forward_propogation(Word1_encoding, Word2_encoding, Word3_encoding)

        loss = self.cross_entropy_loss(target, prediction)

        derivative_of_loss_wrt_softmax = -target / prediction
        derivative_of_softmax_wrt_sigmoid = self.W3

        derivative_of_sigmoid_wrt_W2 = embedding_layer1 + embedding_layer2 + embedding_layer3

        derivative_of_softmax_wrt_W3 = hidden_layer

        derivative_of_loss_wrt_W1 = (derivative_of_loss_wrt_softmax @ derivative_of_softmax_wrt_sigmoid @ self.W21).T @ (Word1_encoding + Word2_encoding + Word3_encoding) + (derivative_of_loss_wrt_softmax @ derivative_of_softmax_wrt_sigmoid @ self.W22).T @ (Word1_encoding + Word2_encoding + Word3_encoding) + (derivative_of_loss_wrt_softmax @ derivative_of_softmax_wrt_sigmoid @ self.W23).T @ (Word1_encoding + Word2_encoding + Word3_encoding)

        derivative_of_loss_wrt_W21 = (derivative_of_loss_wrt_softmax @ derivative_of_softmax_wrt_sigmoid).T @ derivative_of_sigmoid_wrt_W2.T
        derivative_of_loss_wrt_W22 = (derivative_of_loss_wrt_softmax @ derivative_of_softmax_wrt_sigmoid).T @ derivative_of_sigmoid_wrt_W2.T
        derivative_of_loss_wrt_W23 = (derivative_of_loss_wrt_softmax @ derivative_of_softmax_wrt_sigmoid).T @ derivative_of_sigmoid_wrt_W2.T

        derivative_of_loss_wrt_W3 = derivative_of_loss_wrt_softmax.T @ derivative_of_softmax_wrt_W3.T

        derivative_of_loss_wrt_bias1 =  (derivative_of_loss_wrt_softmax @ derivative_of_softmax_wrt_sigmoid).T 

        derivative_of_loss_wrt_bias2 =  derivative_of_loss_wrt_softmax.T

        return derivative_of_loss_wrt_W1, derivative_of_loss_wrt_W21, derivative_of_loss_wrt_W22, derivative_of_loss_wrt_W23, derivative_of_loss_wrt_W3, derivative_of_loss_wrt_bias1, derivative_of_loss_wrt_bias2, loss

network = Network(X, codebook)
# embedding_layer1, embedding_layer2, embedding_layer3, hidden_layer, prediction = network.forward_propogation(Word1_encoding, Word2_encoding, Word3_encoding)
# loss = network.cross_entropy_loss(word4_target, prediction)
derivative_of_loss_wrt_W1, derivative_of_loss_wrt_W21, derivative_of_loss_wrt_W22, derivative_of_loss_wrt_W23, derivative_of_loss_wrt_W3, derivative_of_loss_wrt_bias1, derivative_of_loss_wrt_bias2, loss= network.backward_propagation(Word1_encoding, Word2_encoding, Word3_encoding, word4_target)

print(loss)





