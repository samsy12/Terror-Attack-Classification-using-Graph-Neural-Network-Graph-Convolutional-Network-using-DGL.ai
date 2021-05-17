# Terror-Attack-Classification-using-Graph-Neural-Network-Graph-Convolutional-Network-using-DGL.ai
## Summary
Used Terror Attack dataset from https://linqs-data.soe.ucsc.edu/public/lbc/TerroristRel.tgz in node classification using Graph Convolutional Network.
The project is implemented using Deep Graph Library (DGL.ai) using PyTorch backend.
All necessary installations are included in the notebook itself.
The notebook can be run as it is in Google Colab.
I have separated the original data from the source into seperate CSV files for convenience.

## Details
The ids of the terror attacks in the dataset are URLs. So, I have mapped them to integer ids. I have done the same with the labels. For easy loading, I have seperated the data into seperate CSV files (for further details please read the README file inside Data directory). The features are in the form of an adjacency matrix.
I have used 2 Graph Convolution layers (GraphConv), cross_entropy loss function, Adam optimizer and argmax for prediction of labels.
The model reached a validation accuracy of 80% with 20 epochs. The parameters are provided for reproduction in ./Model/demo.pth.
