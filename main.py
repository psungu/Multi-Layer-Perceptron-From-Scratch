from Network import Network
import numpy as np
import matplotlib.pyplot as plt
from statistics import mean
import pickle


def read_data():

    X = np.load('train_inputs.npy')
    Y = np.load('train_targets.npy')
    codebook =  np.load('vocab.npy')

    Word4 = np.eye(len(codebook))[Y]

    word1_target = X[:,0].reshape(-1)
    Word1_encoding = np.eye(len(codebook))[word1_target]

    word2_target = X[:,1].reshape(-1)
    Word2_encoding = np.eye(len(codebook))[word2_target]

    word3_target = X[:,2].reshape(-1)
    Word3_encoding = np.eye(len(codebook))[word3_target]

    return Word1_encoding, Word2_encoding, Word3_encoding, Word4, codebook



def shuffle_split_dataset(batch_size=1250):

    Word1_encoding, Word2_encoding, Word3_encoding, Word4, codebook = read_data()

    suff = np.arange(Word4.shape[0])
    np.random.shuffle(suff)
    Word1_encoding = np.array_split(Word1_encoding[suff], batch_size, axis=0)
    Word2_encoding = np.array_split(Word2_encoding[suff],batch_size, axis=0)
    Word3_encoding = np.array_split(Word3_encoding[suff],batch_size, axis=0)
    Word4 = np.array_split(Word4[suff], batch_size, axis=0)
    return Word1_encoding, Word2_encoding, Word3_encoding, Word4, codebook


Word1_encoding, Word2_encoding, Word3_encoding, Word4, codebook = shuffle_split_dataset()

epochs = 10
learning_rates = []
result_loss = []
learning_rate = 1e-4
network = Network(codebook)
predictions = []
for epoch in range(epochs):

    loss_values = []

    for i in range(len(Word1_encoding)):
        
        derivative_of_loss_wrt_W1, derivative_of_loss_wrt_W21, derivative_of_loss_wrt_W22, derivative_of_loss_wrt_W23, derivative_of_loss_wrt_W3, derivative_of_loss_wrt_bias1, derivative_of_loss_wrt_bias2, loss= network.backpropagation(Word1_encoding[i], Word2_encoding[i], Word3_encoding[i], Word4[i])
        learning_rates.append(learning_rate)
        loss_values.append(loss)

        network.W1 = network.W1 - derivative_of_loss_wrt_W1 * learning_rate
        network.W21 = network.W21 - derivative_of_loss_wrt_W21 * learning_rate
        network.W22 = network.W22 - derivative_of_loss_wrt_W22 * learning_rate
        network.W23 = network.W23 - derivative_of_loss_wrt_W23 * learning_rate
        network.W3 = network.W3 - derivative_of_loss_wrt_W3 * learning_rate
        network.bias1 = network.bias1 - derivative_of_loss_wrt_bias1 * learning_rate
        network.bias2 = network.bias2 - derivative_of_loss_wrt_bias2 * learning_rate
        learning_rate = learning_rate * (1 / (1 + 1e-4 * epochs))

        embedding_layer1, embedding_layer2, embedding_layer3, hidden_layer, prediction = network.forward_propogation(Word1_encoding[i], Word2_encoding[i], Word3_encoding[i])
        pred = np.argmax(prediction, axis=1)
        predictions.extend(pred)


    result_loss.append(mean(loss_values))




Y = np.load('train_targets.npy')
p = set(predictions) & set(Y)
print(len(p)/len(Y))


model = [network.W1, network.W21, network.W22, network.W23, network.W3, network.bias1, network.bias2]
with open("modelweights.pkl", "wb") as File:
   pickle.dump(model, File)



x = np.arange(len(result_loss))
y = result_loss
plt.plot(x, y)
plt.title('Convergence Curve')
plt.xlabel('Number of Epoch')
plt.ylabel('Loss Values')
plt.show()
