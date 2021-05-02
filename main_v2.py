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


def validation(codebook):

    X = np.load('valid_inputs.npy')
    Y = np.load('valid_targets.npy')

    word1_target = X[:,0].reshape(-1)
    Word1_encoding = np.eye(len(codebook))[word1_target]

    word2_target = X[:,1].reshape(-1)
    Word2_encoding = np.eye(len(codebook))[word2_target]

    word3_target = X[:,2].reshape(-1)
    Word3_encoding = np.eye(len(codebook))[word3_target]
    
    return Word1_encoding, Word2_encoding, Word3_encoding, Y



def shuffle_split_dataset(batch_size=1000):

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


epochs = 15
learning_rates = []
result_loss = []
result_prediction = []
result_actual = []
learning_rate = 0.001
network = Network(codebook)

valid1_encoding, valid2_encoding, valid3_encoding, valid_target = validation(codebook)

for epoch in range(epochs):

    loss_values = []
    predictions = []
    tot = 0
    count_epoch = 0
    
    for i in range(len(Word1_encoding)):
        total = 0
        counter = 0

        derivative_of_loss_wrt_W1, derivative_of_loss_wrt_W2, derivative_of_loss_wrt_W3, derivative_of_loss_wrt_bias1, derivative_of_loss_wrt_bias2, loss = network.backpropagation(Word1_encoding[i], Word2_encoding[i], Word3_encoding[i], Word4[i])
        learning_rates.append(learning_rate)
        loss_values.append(loss)

        network.W1 -= (derivative_of_loss_wrt_W1 ) * learning_rate 
        network.W2 -= (derivative_of_loss_wrt_W2 ) * learning_rate
        network.W3 -= (derivative_of_loss_wrt_W3 ) * learning_rate
        network.bias1 -= derivative_of_loss_wrt_bias1 * learning_rate
        network.bias2 -= derivative_of_loss_wrt_bias2 * learning_rate

        _,_,_, prediction = network.forward_propogation(Word1_encoding[i], Word2_encoding[i], Word3_encoding[i])

        pred = np.argmax(prediction, axis=1)
        actual = np.argmax(Word4[i], axis = 1)

        for j in range(len(actual)):
            total+=1
            if (actual[j] == pred[j]):
                counter+=1


    print("Training Accuracy: {0}".format(counter/total))

    result_prediction.append(counter/total)
    

    _, _, _, prediction = network.forward_propogation(valid1_encoding, valid2_encoding, valid3_encoding)

    pred = np.argmax(prediction, axis=1)

    count = 0

    for j in range(len(valid_target)):
        if (valid_target[j] == pred[j]):
            count+=1

    print("Validation Accuracy: {0}".format(count/len(valid_target)))


    learning_rate = learning_rate * 0.01

    
    result_loss.append(mean(loss_values))




model = [network.W1, network.W2, network.W3, network.bias1, network.bias2]
with open("modelweights.pkl", "wb") as File:
   pickle.dump(model, File)

