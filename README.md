# TweetSentimentAnalysis
An implementation of neural networks to classify twitter tweet's sentiments. 

**How to Use**

1. Specify data path.
2. Call `df = pd.read_csv(path)` to load dataset.

*Note:* dataset should have `column0` - content, `column1` - sentiments

3. Call `data, labels = polishDataSet(df)` to polish the data.
4. Call `train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size = test_split, random_state = 42)`
5. Call `train_data_reduced, vectorizer, reducer = dataReduce(train_data, vectorizer, n_components = 500, ngram = False, ngram_value = 1)`

***vectorizer:*** should be either `'bow'` or `'tfidf'`

***n_components:*** number of features in the final word vector

***ngram:*** set `True` if you want to use n-gram instead of individual words

***ngram_value:*** if `ngram = True`, then set the value of ngram here

6. Call `model = train(train_data_reduced, train_labels, epochs, batchSize, hidden_layer = 50, activation = 'relu', optimizer = 'adam', loss = 'categorical_crossentropy')` to train the neural network.

***epochs:*** set the number of epochs

***batchSize:*** set the batchSize for training

***hidden_layer:*** set the number of neurons in the hidden layer

***activation:*** set the activation function of the hidden layer

***optimizer:*** set the optimizer for the neural network

***loss:*** set the loss function for the neural network

7. Call `test(test_data, test_labels, model, vectorizer, reducer)` for testing the model.





