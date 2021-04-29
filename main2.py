from Network_v2 import Network
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



def shuffle_split_dataset(batch_size=250):

    Word1_encoding, Word2_encoding, Word3_encoding, Word4, codebook = read_data()

    data = np.concatenate([Word1_encoding, Word2_encoding, Word3_encoding, Word4], axis=1)
    np.random.shuffle(data)

    split_data = np.array_split(data, 4, axis=1)
    Word1_encoding = split_data[0]
    Word2_encoding = split_data[1]
    Word3_encoding = split_data[2]
    Word4 = split_data[3]


    Word1_encoding = np.array_split(Word1_encoding, batch_size, axis=0)
    Word2_encoding = np.array_split(Word2_encoding,batch_size, axis=0)
    Word3_encoding = np.array_split(Word3_encoding,batch_size, axis=0)
    Word4 = np.array_split(Word4, batch_size, axis=0)
    return Word1_encoding, Word2_encoding, Word3_encoding, Word4, codebook


Word1_encoding, Word2_encoding, Word3_encoding, Word4, codebook = shuffle_split_dataset()

epochs = 35
learning_rates = []
result_loss = []
result_prediction = []
result_actual = []
learning_rate = 0.01
network = Network(codebook)
Lambda = 1/128

for epoch in range(epochs):

    loss_values = []
    predictions = []
    counter = 0
    total = 0

    for i in range(len(Word1_encoding)):
        
        derivative_of_loss_wrt_W1, derivative_of_loss_wrt_W21, derivative_of_loss_wrt_W22, derivative_of_loss_wrt_W23, derivative_of_loss_wrt_W3, derivative_of_loss_wrt_bias1, derivative_of_loss_wrt_bias2, loss = network.backpropagation(Word1_encoding[i], Word2_encoding[i], Word3_encoding[i], Word4[i])
        learning_rates.append(learning_rate)
        loss_values.append(loss)

        network.W1 -= derivative_of_loss_wrt_W1 * learning_rate
        network.W21 -= derivative_of_loss_wrt_W21 * learning_rate
        network.W22 -= derivative_of_loss_wrt_W22 * learning_rate
        network.W23 -= derivative_of_loss_wrt_W23 * learning_rate

        network.W3 -= derivative_of_loss_wrt_W3 * learning_rate
        network.bias1 -= derivative_of_loss_wrt_bias1 * learning_rate
        network.bias2 -= derivative_of_loss_wrt_bias2 * learning_rate

        _,_,_,_,_,_,_, prediction = network.forward_propogation(Word1_encoding[i], Word2_encoding[i], Word3_encoding[i])

        pred = np.argmax(prediction, axis=1)
        actual = np.argmax(Word4[i], axis = 1)

        for j in range(len(actual)):
            total+=1
            if (actual[j] == pred[j]):
                counter+=1

    learning_rate = learning_rate * 0.01

    
    print(counter/total)
    result_loss.append(mean(loss_values))




model = [network.W1, network.W21, network.W22, network.W23, network.W3, network.bias1, network.bias2]
with open("modelweights_new.pkl", "wb") as File:
   pickle.dump(model, File)



x = np.arange(len(result_loss))
y = result_loss
plt.plot(x, y)
plt.title('Convergence Curve')
plt.xlabel('Number of Epoch')
plt.ylabel('Loss Values')
plt.show()


x = np.arange(len(result_prediction))
y = result_prediction
plt.plot(x, y)
plt.title('Accuracy Curve')
plt.xlabel('Number of Epoch')
plt.ylabel('Accuracy')
plt.show()
