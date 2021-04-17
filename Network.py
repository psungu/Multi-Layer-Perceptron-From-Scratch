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
        # self.W1_initialization = np.random.randn(len(codebook), 16)
        # self.W2_initialization = np.random.randn(48, 128)
        # self.W3_initialization = np.random.randn(128, len(codebook))
        # self.bias1_initialization = np.random.randn(X.shape[0],1)
        # self.bias2_initialization = np.random.randn(X.shape[0],1)

        self.W1_initialization = np.ones([len(codebook), 16])
        self.W2_initialization = np.ones([48, 128])
        self.W3_initialization = np.ones([128, len(codebook)])
        self.bias1_initialization = np.ones([X.shape[0],1])
        self.bias2_initialization = np.ones([X.shape[0],1])

    def sigmoid(self, S):

        return (1 / (1 + np.exp(-S)))

    def softmax(self, S):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(S) / np.sum(np.exp(S), axis=0)

    def forward_propogation(self, Word1_encoding, Word2_encoding, Word3_encoding):

        embedding_layer1 = Word1_encoding @ self.W1_initialization
        embedding_layer2 = Word2_encoding @ self.W1_initialization
        embedding_layer3 = Word3_encoding @ self.W1_initialization
        word_embedding_output = np.concatenate((embedding_layer1, embedding_layer2, embedding_layer3), axis=1)
        hidden_layer = word_embedding_output @ self.W2_initialization + self.bias1_initialization
        hidden_layer_output = self.sigmoid(hidden_layer)
        word_layer = hidden_layer_output @ self.W3_initialization + self.bias2_initialization
        prediction = self.softmax(word_layer)

        return prediction


    def cross_entropy_loss(self, Y, prediction):
	    return -np.sum([Y[i] * np.log(prediction[i]) for i in range(len(prediction))])* 1/len(Y)




network = Network(X, codebook)
prediction = network.forward_propogation(Word1_encoding, Word2_encoding, Word3_encoding)
loss = network.cross_entropy_loss(word4_target, prediction)
print(loss)





