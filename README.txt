Create virtual environment with Python 3.8.7 
And, install the packages provided in requirements

-Requirements

import numpy as np
import pickle
import matplotlib.pyplot as plt
from statistics import mean
from sklearn.manifold import TSNE

-main.py

Main file includes, read training and validation datasets, traning dataset is shuffled
and divided into mini batches, default mini batch number is 250 which provides 1490 
data points in each batch as a batch size. In the case of memory allocation error, you 
may increase the mini batch number, and you would solve the memory allocation error.
However, keeping the batch size as large as possible is suggested.

Weight updates namely training is done in the file, to call the main file from the
terminal, one may use the following line:

python main.py

After 15 second later, you could see the training and validation accuracy from the terminal.
By default 15 epochs are given, to change it one may open the file and change the assignment 
of epochs variable. After training is completed, model weights will be saved to the same path
as a "modelweights.pkl". 


-eval.py

Eval file includes read test dataset, load model, predict, and accuracy calculation methods.
to call the eval file from the terminal, one may use the following line:

python eval.py

In seconds, you will see the test accuracy value, and 3 sample predictions with the input words. 

-tsne.py

Tsne file includes read the codebook, and load the model. Then produce a tsne plot. To call
the tsne file from the terminal,  one may use the following line:

python tsne.py

-Network.py

To train a model with default settings, there is no need to do anything with this file.  
Whether one may want to change the initialization settings, one may open the Network.py file
and change properties of the __init__() function.


There are also main_v2 and Network_v2 files, they are only for experiments and debug the 
main and Network files. Main.py does not concatenate the embedding layers when main_v2.py
uses concatenation, there is not differences between them. Only return variables are changes
on backpropagation algorithm, eval and tsne files are prepared for the Main and Network files
not for the v2s.