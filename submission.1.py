import numpy as np
import pandas as pd

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def model(X, theta):
    return sigmoid(np.dot(X, theta.T))

def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)))
    right = np.multiply(1 - y, np.log(1 - model(X, theta)))
    return np.sum(left - right) / (len(X))  

def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta)- y).ravel()
    #print(grad)
    for j in range(len(theta.ravel())): #for each parmeter
        #print(j)
        term = np.multiply(error, X[:,j])
        #print(term)
        grad[j] = np.sum(term) #/ len(X)
    
    return grad

STOP_ITER = 0  #根据迭代次数停止迭代
STOP_COST = 1  #根据损失值停止迭代
STOP_GRAD = 2  #根据梯度变化停止迭代

def stopCriterion(type, value, threshold):
    #设定三种不同的停止策略
    if type == STOP_ITER:        return value > threshold
    elif type == STOP_COST:      return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:      return np.linalg.norm(value) < threshold

def shuffleData(data):
    np.random.shuffle(data)
    cols = data.shape[1]
    X = data[:, 0:cols-1]
    y = data[:, cols-1:]
    return X, y

# stopType停止策略；thresh预值；alpha学习率
def descent(data, labels, theta, batchSize, stopType, thresh, alpha):
    #梯度下降求解
    
    i = 0 # 迭代次数
    k = 0 # batch
    X = data
    n = len(X)
    y = labels
    grad = np.zeros(theta.shape) # 计算的梯度
    costs = [cost(X, y, theta)] # 损失值

    
    while True:
        #print(k)
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize #取batch数量个数据
        if k >= n: 
            k = 0 
        theta = theta - alpha*grad # 参数更新
        costs.append(cost(X, y, theta)) # 计算新的损失
        i += 1 

        if stopType == STOP_ITER:       
            value = i
        elif stopType == STOP_COST:     
            value = costs
        elif stopType == STOP_GRAD:     
            value = grad
        if stopCriterion(stopType, value, thresh): 
            break
    
    return theta, i-1, costs, grad

# def runExpe(data, theta, batchSize, stopType, thresh, alpha):
#     #import pdb; pdb.set_trace();
#     theta, iter, costs, grad, dur = descent(data, theta, batchSize, stopType, thresh, alpha)
#     name = "Original" if (data[:,1]>2).sum() > 1 else "Scaled"
#     name += " data - learning rate: {} - ".format(alpha)
#     if batchSize==n: 
#         strDescType = "Gradient"
#     elif batchSize==1:  
#         strDescType = "Stochastic"
#     else: 
#         strDescType = "Mini-batch ({})".format(batchSize)
#     name += strDescType + " descent - Stop: "
#     if stopType == STOP_ITER: 
#         strStop = "{} iterations".format(thresh)
#     elif stopType == STOP_COST: 
#         strStop = "costs change < {}".format(thresh)
#     else: 
#         strStop = "gradient norm < {}".format(thresh)
#     name += strStop
#     print ("***{}\nTheta: {} - Iter: {} - Last cost: {:03.2f} - Duration: {:03.2f}s".format(
#         name, theta, iter, costs[-1], dur))
#     fig, ax = plt.subplots(figsize=(12,4))
#     ax.plot(np.arange(len(costs)), costs, 'r')
#     ax.set_xlabel('Iterations')
#     ax.set_ylabel('Cost')
#     ax.set_title(name.upper() + ' - Error vs. Iteration')
#     return theta

def logistic_regression(data, labels, weights, num_epochs, learning_rate): # do not change the heading of the function
    batchSize = len(data)
    stopType = 0
    data = np.insert(data, 0, values=1, axis=1)
    #print(data[:5])
    theta, iter, costs, grad = descent(data, labels, weights, batchSize, stopType, num_epochs, learning_rate)
    print(costs[0],costs[-1])
    #                          descent(data, labels, theta, batchSize, stopType, thresh, alpha)
    print(sum((model(data, theta)- labels).ravel())/len(data))
    return theta
    





data_file ='./asset/a'
raw_data = pd.read_csv(data_file, sep=',')
raw_data.head()
raw_data = pd.read_csv(data_file, sep=',')
labels=raw_data['Label'].values
data=np.stack((raw_data['Col1'].values,raw_data['Col2'].values), axis=-1)

## Fixed Parameters. Please do not change values of these parameters...
weights = np.zeros(3) # We compute the weight for the intercept as well...
num_epochs = 50000
learning_rate = 50e-5
#print(data)
#print(labels)
coefficients=logistic_regression(data, labels, weights, num_epochs, learning_rate)
print(coefficients)