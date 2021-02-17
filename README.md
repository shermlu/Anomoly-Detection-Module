# Anomoly-Detection-Module
This model utilizes an autoencoder to perform anomoly detection. The model was created in keras/tensorflow. I chose an autoencoder because of it's specific usage with regards to anomoly detection. An autoencoder seeks to take an original image and recreate it. In this case, it passes the data without anomolies through an encoder, and passes it through a decoder to replicate the data. This works well with anomoly detection because when you try to recreate the data with anomolies, your model should fail to model the anomoly points. 

In this model, the encoder and decoder parts only have two layers with a batch normalization between each layer. Both the encoder and decoder are the same; the first layer in the encoder is the last layer of the decoder, and the second layer of the encoder is the last layer of the decoder. The goal of an autoencoder is to take an input and to learn and replicate it. The architecture of this model supports that goal. The results of the model are below:


cpc results: 
Confusion matrix 2
| 1617 | 6 |
|------|---|
| 1    | 0 |

Confusion matrix 3
| 1530 | 5 |
|------|---|
| 3    | 0 |

Confusion matrix 4
| 1636 | 4 |
|------|---|
| 1    | 2 |
 
cpm results: 
Confusion matrix 2
| 1613 | 9 |
|------|---|
| 2    | 0 |

Confusion matrix 3
| 1533 | 4 |
|------|---|
| 1    | 0 |

Confusion matrix 4
| 1635 | 4 |
|------|---|
| 4    | 0 |
