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

    H1 = model[1] @ embedding_layer1
    H2 = model[2] @ embedding_layer2
    H3 = model[3] @ embedding_layer3

    hidden_layer = sigmoid(H1+H2+H3 + model[5])
        
    prediction = softmax(model[4] @ hidden_layer + model[6])

    return prediction


def calculate_accuracy(target):
    prediction = predict(model, Word1_encoding, Word2_encoding, Word3_encoding)
    predict_index = np.argmax(prediction, axis=0)
    counter = 0
    for j in range(len(target)):
        if (target[j] == predict_index[j]):
            counter+=1

    return counter/len(target)


def predict_given_sequence(X1, X2, X3, codebook, model):
    Word1_encoding = np.zeros([1,250])
    Word2_encoding = np.zeros([1,250])
    Word3_encoding = np.zeros([1,250])

    X1_ind=np.where(codebook == X1)[0][0]
    X2_ind=np.where(codebook == X2)[0][0]
    X3_ind=np.where(codebook == X3)[0][0]

    Word1_encoding[:,X1_ind] = 1
    Word2_encoding[:,X2_ind] = 1
    Word3_encoding[:,X3_ind] = 1
    
    prediction = predict(model, Word1_encoding, Word2_encoding, Word3_encoding)

    result_ind = np.argmax(prediction)

    return codebook[result_ind]

    
accuracy = calculate_accuracy(Word4)
print("Test accuracy is: {0}".format(accuracy))

text_result1 = predict_given_sequence("city", "of", "new", codebook, model)
print("city of new: {0}".format(text_result1))

text_result2 = predict_given_sequence("life", "in", "the", codebook, model)
print("life in the: {0}".format(text_result2))

text_result3 = predict_given_sequence("he", "is", "the", codebook, model)
print("he is the: {0}".format(text_result3))
