import numpy as np
import pickle

def read_data():

    X = np.load('test_inputs.npy')
    Y = np.load('test_targets.npy')
    codebook =  np.load('vocab.npy')

    word1_target = X[:,0].reshape(-1)
    Word1_encoding = np.eye(len(codebook))[word1_target]

    word2_target = X[:,1].reshape(-1)
    Word2_encoding = np.eye(len(codebook))[word2_target]

    word3_target = X[:,2].reshape(-1)
    Word3_encoding = np.eye(len(codebook))[word3_target]

    return Word1_encoding, Word2_encoding, Word3_encoding, Y, codebook


def load_model(path):
    file = open("modelweights.pkl",'rb')
    object_file = pickle.load(file)
    return object_file


Word1_encoding, Word2_encoding, Word3_encoding, Word4, codebook = read_data()
model = load_model("./")


def softmax(Z):
    expZ = np.exp(Z - np.max(Z))
    return expZ / expZ.sum(axis=0, keepdims=True)


def sigmoid(S):

    return (1 / (1 + np.exp(-S)))

def predict(model, Word1_encoding, Word2_encoding, Word3_encoding):
    embedding_layer1 = model[0] @ Word1_encoding.T
    embedding_layer2 = model[0] @ Word2_encoding.T 
    embedding_layer3 = model[0] @ Word3_encoding.T

    hidden_layer_1 = sigmoid(model[1] @ embedding_layer1  + model[5])
    hidden_layer_2 = sigmoid(model[2] @ embedding_layer2  + model[5])
    hidden_layer_3 = sigmoid(model[3] @ embedding_layer3  + model[5])
    hidden_layer = hidden_layer_1 + hidden_layer_2 + hidden_layer_3
    
    prediction = softmax(model[4] @ hidden_layer + model[6])

    return prediction


def calculate_accuracy(target):
    prediction = predict(model, Word1_encoding, Word2_encoding, Word3_encoding)
    predict_index = np.argmax(prediction, axis=1)

    correct_classified = set(predict_index) & set(target)
    return (len(correct_classified)/len(target))


accuracy = calculate_accuracy(Word4)
print(accuracy)