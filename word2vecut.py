# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 23:13:39 2019

@author: Utsav Tiwari
"""



import matplotlib.pyplot as plt
import tensorflow as tf

#def initialize_wrd_emb(vocab_size, emb_size):
#    """
#    vocab_size: int. vocabulary size of your corpus or training data
#    emb_size: int. word embedding size. How many dimensions to represent each vocabulary
#    """
#    WRD_EMB = np.random.randn(vocab_size, emb_size) * 0.01
#    return WRD_EMB
#
#def initialize_dense(input_size, output_size):
#    """
#    input_size: int. size of the input to the dense layer
#    output_szie: int. size of the output out of the dense layer
#    """
#    W = np.random.randn(output_size, input_size) * 0.01
#    return W
#
#def initialize_parameters(vocab_size, emb_size):
#    """
#    initialize all the trianing parameters
#    """
#    WRD_EMB = initialize_wrd_emb(vocab_size, emb_size)
#    W = initialize_dense(emb_size, vocab_size)
#    
#    parameters = {}
#    parameters['WRD_EMB'] = WRD_EMB
#    parameters['W'] = W
#    
#    return parameters
#
#
#
#
#
#
#
#
#def ind_to_word_vecs(inds, parameters):
#    """
#    inds: numpy array. shape: (1, m)
#    parameters: dict. weights to be trained
#    """
#    m = inds.shape[1]
#    WRD_EMB = parameters['WRD_EMB']
#    word_vec = WRD_EMB[inds.flatten(), :].T
#    
#    assert(word_vec.shape == (WRD_EMB.shape[1], m))
#    
#    return word_vec
#
#def linear_dense(word_vec, parameters):
#    """
#    word_vec: numpy array. shape: (emb_size, m)
#    parameters: dict. weights to be trained
#    """
#    m = word_vec.shape[1]
#    W = parameters['W']
#    Z = np.dot(W, word_vec)
#    
#    assert(Z.shape == (W.shape[0], m))
#    
#    return W, Z
#
#def softmax(Z):
#    """
#    Z: output out of the dense layer. shape: (vocab_size, m)
#    """
#    softmax_out = np.divide(np.exp(Z), np.sum(np.exp(Z), axis=0, keepdims=True) + 0.001)
#    
#    assert(softmax_out.shape == Z.shape)
#
#    return softmax_out
#
#def forward_propagation(inds, parameters):
#    word_vec = ind_to_word_vecs(inds, parameters)
#    W, Z = linear_dense(word_vec, parameters)
#    softmax_out = softmax(Z)
#    
#    caches = {}
#    caches['inds'] = inds
#    caches['word_vec'] = word_vec
#    caches['W'] = W
#    caches['Z'] = Z
#    
#    return softmax_out, caches
#
#
#
#
#
#
#
#
#def cross_entropy(softmax_out, Y):
#    """
#    softmax_out: output out of softmax. shape: (vocab_size, m)
#    """
#    m = softmax_out.shape[1]
#    cost = -(1 / m) * np.sum(np.sum(Y * np.log(softmax_out + 0.001), axis=0, keepdims=True), axis=1)
#    return cost
#
#
#
#def softmax_backward(Y, softmax_out):
#    """
#    Y: labels of training data. shape: (vocab_size, m)
#    softmax_out: output out of softmax. shape: (vocab_size, m)
#    """
#    dL_dZ = softmax_out - Y
#    
#    assert(dL_dZ.shape == softmax_out.shape)
#    return dL_dZ
#
#def dense_backward(dL_dZ, caches):
#    """
#    dL_dZ: shape: (vocab_size, m)
#    caches: dict. results from each steps of forward propagation
#    """
#    W = caches['W']
#    word_vec = caches['word_vec']
#    m = word_vec.shape[1]
#    
#    dL_dW = (1 / m) * np.dot(dL_dZ, word_vec.T)
#    dL_dword_vec = np.dot(W.T, dL_dZ)
#
#    assert(W.shape == dL_dW.shape)
#    assert(word_vec.shape == dL_dword_vec.shape)
#    
#    return dL_dW, dL_dword_vec
#
#def backward_propagation(Y, softmax_out, caches):
#    dL_dZ = softmax_backward(Y, softmax_out)
#    dL_dW, dL_dword_vec = dense_backward(dL_dZ, caches)
#    
#    gradients = dict()
#    gradients['dL_dZ'] = dL_dZ
#    gradients['dL_dW'] = dL_dW
#    gradients['dL_dword_vec'] = dL_dword_vec
#    
#    return gradients
#
#def update_parameters(parameters, caches, gradients, learning_rate):
#    vocab_size, emb_size = parameters['WRD_EMB'].shape
#    inds = caches['inds']
#    WRD_EMB = parameters['WRD_EMB']
#    dL_dword_vec = gradients['dL_dword_vec']
#    m = inds.shape[-1]
#    
#    WRD_EMB[inds.flatten(), :] -= dL_dword_vec.T * learning_rate
#
#    parameters['W'] -= learning_rate * gradients['dL_dW']
#    
#    
#    
#    
#    
#    
#    
#    
#    
#def skipgram_model_training(X, Y, vocab_size, emb_size, learning_rate, epochs, batch_size, parameters=None, print_cost=True, plot_cost=True):
#    """
#    X: Input word indices. shape: (1, m)
#    Y: One-hot encodeing of output word indices. shape: (vocab_size, m)
#    vocab_size: vocabulary size of your corpus or training data
#    emb_size: word embedding size. How many dimensions to represent each vocabulary
#    learning_rate: alaph in the weight update formula
#    epochs: how many epochs to train the model
#    batch_size: size of mini batch
#    parameters: pre-trained or pre-initialized parameters
#    print_cost: whether or not to print costs during the training process
#    """
#    costs = []
#    m = X.shape[1]
#    
#    if parameters is None:
#        parameters = initialize_parameters(vocab_size, emb_size)
#    
#    for epoch in range(epochs):
#        epoch_cost = 0
#        j=0
#        batch_inds = list(range(0, m, batch_size))
#        np.random.shuffle(batch_inds)
#        for i in batch_inds:
#            X_batch = X[:, i:i+batch_size]
#            
#            Y_batch = Y[:, i:i+batch_size]
#            
#
#            softmax_out, caches = forward_propagation(X_batch, parameters)
#            gradients = backward_propagation(Y_batch, softmax_out, caches)
#            update_parameters(parameters, caches, gradients, learning_rate)
#            cost = cross_entropy(softmax_out, Y_batch)
#            epoch_cost += np.squeeze(cost)
#        print(epoch_cost)
#        costs.append(epoch_cost)
#        if print_cost and epoch % (epochs // 500) == 0:
#            print("Cost after epoch {}: {}".format(epoch, epoch_cost))
#        if epoch % (epochs // 100) == 0:
#            learning_rate *= 0.98
#            
#            
#    if plot_cost:
#        plt.plot(np.arange(epochs), costs)
#        plt.xlabel('# of epochs')
#        plt.ylabel('cost')
#    return parameters
#
#
import re
import numpy as np

# function to convert numbers to one hot vectors
def to_one_hot(data_point_index, vocab_size):
    temp = np.zeros(vocab_size)
    
    temp[data_point_index] = 1
    
    return temp



def tokenize(text):
    pattern = re.compile(r'[A-Za-z]+[\w^\']*|[\w^\']*[A-Za-z]+[\w^\']*')
    return pattern.findall(text.lower())


def mapping(tokens):
    ind_to_word=dict()
    word_to_ind=dict()
    
    for i,t in enumerate(set(tokens)):
        word_to_ind[t]=i;
        ind_to_word[i]=t;
        
    return ind_to_word,word_to_ind
def gtd(tokens,word_to_ind,window_size):
    N=len(tokens)
    X,Y=[],[]
    for i in range(N):
        nbr=list(range(max(0,i-window_size),i))+list(range(i+1,min(N,i+window_size+1)))
        
        for j in nbr:
            X.append(word_to_ind[tokens[i]])
            Y.append(word_to_ind[tokens[j]])
    X = np.array(X)
    X = np.expand_dims(X, axis=0)
    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=0)

    return X, Y

strg = "The meaning of unity in diversity is the existence of oneness even after various differences. India is a best example for this concept of unity in diversity. We can see very clearly here that people of different religions, creeds, castes, languages, cultures, lifestyle, dressing sense, faith in God, rituals of worship, etc live together with harmony under one roof means on one land of India."\
       "People living in India are the children of one mother whom we call Mother India. India is a vast and most populated country of the world where people of different religions Hinduism, Buddhism, Islam, Sikhism, Jainism, Christianity and Parsees live together but everyone believes in one theory of Dharma and Karma. People here are god fearing in nature and believe in purification of soul, rebirth, salvation, luxury of heaven and punishments of hell. People here celebrate their festivals (Holi, Diwali, Id, Christmas, Good Friday, Mahavir Jayanti, Buddha Jayanti, etc) very peacefully without harming other religious people." \
      "beating the stock market is a loser's game."

tokens=tokenize(strg)

ind_to_word,word_to_ind=mapping(tokens)


X,Y=gtd(tokens,word_to_ind,window_size=3)

vocab_size=len(ind_to_word)

m=Y.shape[1]

Y_onehot=np.zeros((vocab_size,m))

Y_onehot[Y.flatten(),np.arange(m)]=1


x_train = [] # input word
y_train = [] # output word
for p in X[0,:]:
    x_train.append(to_one_hot(p, vocab_size))
    
for q in Y[0,:]:
    y_train.append(to_one_hot(q, vocab_size))   
# convert them to numpy arrays
x_train = np.asarray(x_train)
y_train = np.asarray(y_train)

x = tf.placeholder(tf.float32, shape=(None, vocab_size))
y_label = tf.placeholder(tf.float32, shape=(None, vocab_size))

EMBEDDING_DIM = 10 # you can choose your own number
W1 = tf.Variable(tf.random_normal([vocab_size, EMBEDDING_DIM]))
b1 = tf.Variable(tf.random_normal([EMBEDDING_DIM])) #bias

hidden_representation = tf.add(tf.matmul(x,W1), b1)


W2 = tf.Variable(tf.random_normal([EMBEDDING_DIM, vocab_size]))
b2 = tf.Variable(tf.random_normal([vocab_size]))
prediction = tf.nn.softmax(tf.add( tf.matmul(hidden_representation, W2), b2))


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init) #make sure you do this!
# define the loss function:
cross_entropy_loss = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(prediction), reduction_indices=[1]))
# define the training step:
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy_loss)
n_iters = 10000
# train for n_iter iterations
for _ in range(n_iters):
    sess.run(train_step, feed_dict={x: x_train, y_label: y_train})
    print('loss is : ', sess.run(cross_entropy_loss, feed_dict={x: x_train, y_label: y_train}))

vectors=sess.run(W1+b1)


print(vectors[ word_to_ind['country'] ])


def euclidean_dist(vec1, vec2):
    return np.sqrt(np.sum((vec1-vec2)**2))
def find_closest(word_index, vectors):
    min_dist = 10000 # to act like positive infinity
    min_index = -1
    query_vector = vectors[word_index]
    for index, vector in enumerate(vectors):
        if euclidean_dist(vector, query_vector) < min_dist and not np.array_equal(vector, query_vector):
            min_dist = euclidean_dist(vector, query_vector)
            min_index = index
    return min_index



print(ind_to_word[find_closest(word_to_ind['india'], vectors)])
print(ind_to_word[find_closest(word_to_ind['existence'], vectors)])
print(ind_to_word[find_closest(word_to_ind['populated'], vectors)])



from sklearn.manifold import TSNE
model = TSNE(n_components=2, random_state=0)
np.set_printoptions(suppress=True)
vectors = model.fit_transform(vectors)

from sklearn import preprocessing
normalizer = preprocessing.Normalizer()
vectors =  normalizer.fit_transform(vectors, 'l2')



import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for word in set(tokens):
    print(word, vectors[word_to_ind[word]][1])
    ax.annotate(word, (vectors[word_to_ind[word]][0],vectors[word_to_ind[word]][1] ))
plt.show()







      
            
            
            
        
        
        
        