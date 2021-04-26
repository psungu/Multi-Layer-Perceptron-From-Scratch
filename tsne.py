import numpy as np
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

def load_model(path):
    file = open("modelweights.pkl",'rb')
    object_file = pickle.load(file)
    return object_file


Word1_encoding, Word2_encoding, Word3_encoding, Word4, codebook = read_data()
model = load_model("./")

embedding_layer1 = model[0] @ Word1_encoding.T
embedding_layer2 = model[0] @ Word2_encoding.T 
embedding_layer3 = model[0] @ Word3_encoding.T

