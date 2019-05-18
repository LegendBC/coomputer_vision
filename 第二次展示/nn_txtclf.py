# %% 1
# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from pprint import pprint

Activation = input('input your activation function(RELU,tanh,LEAKRELU):')
decaypolicy = input('input your learning rate decay policy(fix,step,inv,exp,multistep,poly):')
dropoutflagIn = input('need dropout(yes or no):')
maxfeatures = input('maxfeatures:')

if dropoutflagIn == 'yes':
    dropoutflag = True
else:
    dropoutflag = False

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']

newsgroups_train = fetch_20newsgroups(subset='train',  categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test',  categories=categories)

# pprint(newsgroups_train.data[0])

num_train = len(newsgroups_train.data)
num_test  = len(newsgroups_test.data)

vectorizer = TfidfVectorizer(max_features=int(maxfeatures))

X = vectorizer.fit_transform( newsgroups_train.data + newsgroups_test.data )
X_train = X[0:num_train, :]
X_test = X[num_train:num_train+num_test,:]

Y_train = newsgroups_train.target
Y_test = newsgroups_test.target

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

# %% 4
# Helper function to plot a decision boundary.
# If you don't fully understand this function don't worry, it just generates the contour plot below.
def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# %% 7
# Helper function to evaluate the total loss on the dataset

def calculate_loss(model, X, y):
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4'],
    #正向传播，计算预测值
    z1 = X.dot(W1) + b1
    if Activation=='RELU':
        a1=np.maximum(z1,0)
    if Activation=='tanh':
        a1 = np.tanh(z1)
    # if Activation=='LEAKRELU':
    if Activation=='LEAKRELU':
        a1 = np.zeros_like(z1)
        for i, idx in enumerate(z1):
            for j, idy in enumerate(idx):
                if idy > 0:
                    a1[i][j] = idy
                else:
                    a1[i][j] = idy/2


    
    
    z2 = a1.dot(W2) + b2
    if Activation=='RELU':
        a2=np.maximum(z2,0)
    if Activation=='tanh':
        a2 = np.tanh(z2)
    if Activation=='LEAKRELU':
        a2 = np.zeros_like(z2)
        for i, idx in enumerate(z2):
            for j, idy in enumerate(idx):
                if idy > 0:
                    a2[i][j] = idy
                else:
                    a2[i][j] = idy/2
        
    z3 = a2.dot(W3) + b3
    if Activation=='RELU':
        a3=np.maximum(z3,0)
    if Activation=='tanh':
        a3= np.tanh(z3)
    if Activation=='LEAKRELU':
        a3 = np.zeros_like(z3)
        for i, idx in enumerate(z3):
            for j, idy in enumerate(idx):
                if idy > 0:
                    a3[i][j] = idy
                else:
                    a3[i][j] = idy/2

    
    z4 = a3.dot(W4) + b4
    exp_scores = np.exp(z4)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 计算损失
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    #在损失上加上正则项（可选）
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2) + np.square(W3)) + np.sum(np.square(W4)))
    return 1./num_examples * data_loss

def caculateActivationBp(x):
    res = x
    if Activation == 'RELU':
        for i, idx in enumerate(x):
            for j, jdx in enumerate(idx):
                if x[i][j] > 0:
                    res[i][j] = 1
                else: 
                    res[i][j] = 0
    if  Activation == 'tanh':
        res = (1 - np.power(x, 2))
    if Activation == 'LEAKRELU':
        for i, idx in enumerate(x):
            for j, jdx in enumerate(idx):
                if x[i][j] > 0:
                    res[i][j] = 1
                else: 
                    res[i][j] = 1 / 2
    return res 
            
def learning_rate_model(type):
    if type=='fix':
        return base_lr*np.linspace(1,1,epochs).astype(int)
    
    if type=='step':
        return base_lr * gamma**((iter / stepsize).astype(int))
    
    if type=='exp':
        return base_lr * gamma**iter
    
    if type=='inv':
        return base_lr * (1 + gamma * iter)**(- power)
    
    if type=='multistep':
        return base_lr * gamma**((iter / stepvalue).astype(int))
    
    if type=='poly':
        return base_lr*(1 - iter/epochs)**(power)
    
    # for j,iter_n in enumerate(iter):
    #     value=gamma * (iter - stepsize)
    #     iter_sigmod=base_lr*( 1/(1 + np.exp(-value)))
    #     return iter_sigmod
        
    return 0

# %% 8
# Helper function to predict an output (0 or 1)

def train_predict(model, x):
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4'],
    #正向传播，计算预测值
    z1 = x.dot(W1) + b1
    # z1 = X.dot(W1) + b1
    drop1 = (np.random.rand(z1.shape[0],z1.shape[1]) < p)/p
    z1 *=drop1
    if Activation=='RELU':
        a1=np.maximum(z1,0)
    if Activation=='tanh':
        a1 = np.tanh(z1)
    if Activation=='LEAKRELU':
        a1 = np.zeros_like(z1)
        for i, idx in enumerate(z1):
            for j, idy in enumerate(idx):
                if idy > 0:
                    a1[i][j] = idy
                else:
                    a1[i][j] = idy/2


    
    
    z2 = a1.dot(W2) + b2
    drop2 = (np.random.rand(z2.shape[0],z2.shape[1]) < p)/p
    z2 *=drop2
    if Activation=='RELU':
        a2=np.maximum(z2,0)
    if Activation=='tanh':
        a2 = np.tanh(z2)
    if Activation=='LEAKRELU':
        a2 = np.zeros_like(z2)
        for i, idx in enumerate(z2):
            for j, idy in enumerate(idx):
                if idy > 0:
                    a2[i][j] = idy
                else:
                    a2[i][j] = idy/2
        
    z3 = a2.dot(W3) + b3
    drop3 = (np.random.rand(z3.shape[0],z3.shape[1]) < p)/p
    z3 *=drop3
    if Activation=='RELU':
        a3=np.maximum(z3,0)
    if Activation=='tanh':
        a3 = np.tanh(z3)
    if Activation=='LEAKRELU':
        a3 = np.zeros_like(z3)
        for i, idx in enumerate(z3):
            for j, idy in enumerate(idx):
                if idy > 0:
                    a3[i][j] = idy
                else:
                    a3[i][j] = idy/2
    
    z4 = a3.dot(W4) + b4
    exp_scores = np.exp(z4)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs)

def test_predict(model, x):
    W1, b1, W2, b2, W3, b3, W4, b4 = model['W1'], model['b1'], model['W2'], model['b2'], model['W3'], model['b3'], model['W4'], model['b4'],
    #正向传播，计算预测值
    z1 = x.dot(W1) + b1
    # z1 = X.dot(W1) + b1
    if Activation=='RELU':
        a1=np.maximum(z1,0)
    if Activation=='tanh':
        a1 = np.tanh(z1)
    if Activation=='LEAKRELU':
        a1 = np.zeros_like(z1)
        for i, idx in enumerate(z1):
            for j, idy in enumerate(idx):
                if idy > 0:
                    a1[i][j] = idy
                else:
                    a1[i][j] = idy/2


    
    
    z2 = a1.dot(W2) + b2
    if Activation=='RELU':
        a2=np.maximum(z2,0)
    if Activation=='tanh':
        a2 = np.tanh(z2)
    if Activation=='LEAKRELU':
        a2 = np.zeros_like(z2)
        for i, idx in enumerate(z2):
            for j, idy in enumerate(idx):
                if idy > 0:
                    a2[i][j] = idy
                else:
                    a2[i][j] = idy/2
        
    z3 = a2.dot(W3) + b3
    if Activation=='RELU':
        a3=np.maximum(z3,0)
    if Activation=='tanh':
        a3 = np.tanh(z3)
    if Activation=='LEAKRELU':
        a3 = np.zeros_like(z3)
        for i, idx in enumerate(z3):
            for j, idy in enumerate(idx):
                if idy > 0:
                    a3[i][j] = idy
                else:
                    a3[i][j] = idy/2
    
    z4 = a3.dot(W4) + b4
    exp_scores = np.exp(z4)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    return np.argmax(probs)

def dlrelu(x, alpha=.01):
    a1 = np.zeros_like(x)
    for i, idx in enumerate(x):
        for j, idy in enumerate(idx):
            if idy > 0:
                a1[i][j] = 1
            else:
                a1[i][j] = alpha
    return a1

# %% 16
# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations

def build_model(X, y, nn_input_dim, epsilon, reg_lambda,dropOutFlag, num_passes=20000, print_loss=False):
    global best_test_num
    global best_model
     # 用随机值初始化参数。我们需要学习这些参数
    np.random.seed(1)
    W1 = np.random.randn(nn_input_dim[0], nn_hdim) / np.sqrt(nn_input_dim[0])
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_hdim))
    W3 = np.random.randn(nn_hdim, nn_hdim) / np.sqrt(nn_hdim)
    b3 = np.zeros((1, nn_hdim))
    W4 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b4 = np.zeros((1, nn_output_dim))

    # 这是我们最终要返回的数据
    model = {}

    # 梯度下降
    for index in range(0, num_passes):

        # # 正向传播
        # z1 = X.dot(W1) + b1
        # a1 = np.tanh(z1)
        # z2 = a1.dot(W2) + b2
        # a2 = np.tanh(z2)
        # z3 = a2.dot(W3) + b3
        # a3 = np.tanh(z3)
        # z4 = a3.dot(W4) + b4
        # exp_scores = np.exp(z4)
        # probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        z1 = X.dot(W1) + b1
        if dropOutFlag:
            drop1 = (np.random.rand(z1.shape[0],z1.shape[1]) < p)/p
        else:
            drop1 = 1
        z1 *=drop1
        if Activation=='RELU':
            a1=np.maximum(z1,0)
        if Activation=='tanh':
            a1 = np.tanh(z1)
        if Activation=='LEAKRELU':
            a1 = np.zeros_like(z1)
            for i, idx in enumerate(z1):
                for j, idy in enumerate(idx):
                    if idy > 0:
                        a1[i][j] = idy
                    else:
                        a1[i][j] = idy/2


        
        
        z2 = a1.dot(W2) + b2
        if dropOutFlag:
            drop2 = (np.random.rand(z2.shape[0],z2.shape[1]) < p)/p
        else:
            drop2 = 1
        z2 *=drop2
        if Activation=='RELU':
            a2=np.maximum(z2,0)
        if Activation=='tanh':
            a2 = np.tanh(z2)
        if Activation=='LEAKRELU':
            a2 = np.zeros_like(z2)
            for i, idx in enumerate(z2):
                for j, idy in enumerate(idx):
                    if idy > 0:
                        a2[i][j] = idy
                    else:
                        a2[i][j] = idy/2
            
        z3 = a2.dot(W3) + b3
        if dropOutFlag:
            drop3 = (np.random.rand(z3.shape[0],z3.shape[1]) < p)/p
        else:
            drop3 = 1
        z3 *=drop3
        if Activation=='RELU':
            a3=np.maximum(z3,0)
        if Activation=='tanh':
            a3 = np.tanh(z3)
        if Activation=='LEAKRELU':
            a3 = np.zeros_like(z3)
            for i, idx in enumerate(z3):
                for j, idy in enumerate(idx):
                    if idy > 0:
                        a3[i][j] = idy
                    else:
                        a3[i][j] = idy/2
        
        z4 = a3.dot(W4) + b4
        exp_scores = np.exp(z4)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


        # 反向传播
        delta4 = probs
        delta4[range(num_examples), y] -= 1
        dW4 = (a3.T).dot(delta4)
        db4 = np.sum(delta4, axis=0, keepdims=True)
        # delta3 = delta4.dot(W4.T) * caculateActivationBp(a3)
        delta3 = delta4.dot(W4.T) * caculateActivationBp(a3) * drop3
        dW3 = a2.T.dot(delta3)
        db3 = np.sum(delta3, axis=0)
        # delta2 = delta3.dot(W3.T) * caculateActivationBp(a2)
        delta2 = delta3.dot(W3.T) * caculateActivationBp(a2) * drop2
        dW2 = a1.T.dot(delta2)
        db2 = np.sum(delta2, axis=0)
        # delta1 = delta2.dot(W2.T) * caculateActivationBp(a1)
        delta1 = delta2.dot(W2.T) * caculateActivationBp(a1) * drop1
        dW1 = X.T.dot(delta1)
        db1 = np.sum(delta1, axis=0)

        # 添加正则项 (b1 和 b2 没有正则项)
        dW4 += reg_lambda * W4
        dW3 += reg_lambda * W3
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # 梯度下降更新参数
        # W1 += -epsilon * dW1
        # b1 += -epsilon * db1
        # W2 += -epsilon * dW2
        # b2 += -epsilon * db2
        # W3 += -epsilon * dW3
        # b3 += -epsilon * db3
        # W4 += -epsilon * dW4
        # b4 += -epsilon * db4

        # 梯度下降更新参数
        W1 += -epsilon[index] * dW1
        b1 += -epsilon[index] * db1
        W2 += -epsilon[index] * dW2
        b2 += -epsilon[index] * db2
        W3 += -epsilon[index] * dW3
        b3 += -epsilon[index] * db3
        W4 += -epsilon[index] * dW4
        b4 += -epsilon[index] * db4

        # 为模型分配新的参数
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2, 'W3': W3, 'b3': b3, 'W4': W4, 'b4': b4 }

        # 选择性地打印损失
        # 这种做法很奢侈，因为我们用的是整个数据集，所以我们不想太频繁地这样做
        # if print_loss and i % 100 == 0:
        if print_loss and index % 1000 ==0:
            n_correct = 0
            n_train = X_train.shape[0]
            for n in range(n_train):
                x = X_train[n,:]
                yp = train_predict(model, x)
                if yp == Y_train[n]:
                    n_correct += 1.0
            #print('train_Accuracy %f = %d / %d'%(n_correct/n_train, int(n_correct), n_train))
            n_correct = 0
            n_test = X_test.shape[0]
            for n in range(n_test):
                x = X_test[n,:]
                yp = test_predict(model, x)
                if yp == Y_test[n]:
                    n_correct += 1.0
                if n_correct > best_test_num: 
                    best_test_num = n_correct
                    best_model = model
            #print('test_Accuracy %f = %d / %d'%(n_correct/n_test, int(n_correct), n_test))
            print (index, calculate_loss(model, X, y))

    return model



# %% 17
# Build a model with a 3-dimensional hidden layer



num_examples, input_dim = X_train.shape
nn_hdim_list = [3, 5, 7, 10, 12, 15]
nn_output_dim = 4
# epsilon = 0.0001
reg_lambda = 0.01
epochs = 10001
best_test_num = 0
best_model = 0

max_iter = 50000 # the maximum number of iterations
m=np.linspace(1,max_iter,max_iter)
iter=m.astype(int)
iter_sigmod=m

base_lr=0.0001
gamma=0.0001
stepsize=5000
stepvalue=10000
power=0.75
epsilon=learning_rate_model(decaypolicy)

p=0.5
n_correct = 0
n_test = X_test.shape[0]

for i in range(len(nn_hdim_list)):
    nn_hdim = nn_hdim_list[i]
    best_test_num = 0
    best_model = 0
    model = build_model(X_train, Y_train, [input_dim,16,8,4], epsilon, reg_lambda, dropoutflag, epochs, print_loss=True)
    print('channel %d best result%d, %f'%(nn_hdim_list[i], best_test_num, best_test_num / n_test))

for n in range(n_test):
    x = X_test[n,:]
    # yp = test_predict(model, x)
    yp = test_predict(best_model, x)

    if yp == Y_test[n]:
        n_correct += 1.0

print('Accuracy %f = %d / %d'%(n_correct/n_test, int(n_correct), n_test) )

