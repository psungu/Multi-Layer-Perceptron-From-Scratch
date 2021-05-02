import numpy as np
import pickle
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def load_model(path):
    file = open("modelweights.pkl",'rb')
    object_file = pickle.load(file)
    return object_file


codebook = np.load('vocab.npy')
model = load_model("./")


word_encoding = np.identity(250)


def tsne_plot(word_encoding, codebook, model):
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(word_encoding @ model[0].T)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(codebook[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()



tsne_plot(word_encoding, codebook, model)