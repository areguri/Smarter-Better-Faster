import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from heapq import *
import random
import copy
import math
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
import ast

def initialize_parameters_deep(layer_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
      parameters['W'+str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1])*0.01
      parameters['b'+str(l)] = np.zeros((layer_dims[l], 1))
    return parameters

def initialize_parameters_he(layers_dims):
    np.random.seed(3)
    parameters = {}
    L = len(layers_dims)
     
    for l in range(1, L ):
      parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l-1]) * np.sqrt(2/layers_dims[l-1])
      parameters['b' + str(l)] = np.zeros((layers_dims[l], 1))
        
    return parameters

def linear_forward(A, W, b):
  Z = np.dot(W, A) + b
  cache = (A, W, b)
  return Z, cache

def identity(z):
  s = z
  cache = (z)
  return s, cache

def tanh(x):
  cache = (x)
  t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
  return t, cache 

def d_tanh(x):
  t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
  dt=1-t**2
  return dt

def relu(x):
  cache = (x)
  return np.maximum(0.01*x, x), cache

def linear_activation_forward(A_prev, W, b, activation):
  if activation == "identity":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = identity(Z)
  elif activation == "relu":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = relu(Z)
  elif activation == "tanh":
      Z, linear_cache = linear_forward(A_prev, W, b)
      A, activation_cache = tanh(Z)
  cache = (linear_cache, activation_cache)
  return A, cache

def implement_dropout(A, dropout_layers, l):
  for i in range(A.shape[0]):
    if random.random() < dropout_layers[l]:
      A[i,:] = 0
  return A

def L_model_forward(X, parameters, dropout=False, dropout_layers=[], layers_dims=[]):
    caches = []
    A = X
    L = len(layers_dims) - 1
    #print("Layers dims is ", layers_dims)
    for l in range(1, L):
      A_prev = A 
      A, cache = linear_activation_forward(A_prev, parameters["W"+str(l)], parameters["b"+str(l)], "relu")
      caches.append(cache)
      if dropout:
        A = implement_dropout(A, dropout_layers, l)

    AL, cache = linear_activation_forward(A, parameters["W"+str(L)], parameters["b"+str(L)], "identity")
    caches.append(cache)
    return AL, caches

def compute_cost(AL, Y):
  m = Y.shape[1]
  cost = 0
  for i in range(m):
    cost = cost + (Y[0][i] - AL[0][i])*(Y[0][i] - AL[0][i])
  cost = cost/m
  return cost

def linear_backward(dZ, cache):
    A_prev, W, b = cache
    m = A_prev.shape[1]
    dW = (1/m)*np.dot(dZ, A_prev.T)
    db = (1/m)*np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def relu_backward(dA, cache):
  #print("Da is ", dA)
  #print("cache is ", cache)
  return dA*dReLU(cache)

def tanh_backward(dA, cache):
  #print("Da is ", dA)
  #print("cache is ", cache)
  return dA*d_tanh(cache)

def dReLU(x, alpha=.01):
  return np.where(x>=0, 1, alpha)


def linear_diff_backward(dA, cache):
  return dA

def linear_activation_backward(dA, cache, activation):
    #print("DDAAAA, cache and activation is ", dA, cache, activation)
    linear_cache, activation_cache = cache
    #print("came here")
    if activation == "relu":
        #print("relu backward")
        dZ = relu_backward(dA, activation_cache)
        #print("DZrelu is ", dZ)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        #print(dA_prev, dW, db, " are the next layer results ")
    elif activation == "linear":
        #print("DAAAAAAAAAAAAAAAAAAAAAAAAA is ", dA)
        dZ = dA
        #print("DZnext is ", dZ)
        #print("okay ")
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        #print(dA_prev, dW, db, " are the first layer results ")
    if activation == "tanh":
        #print("relu backward")
        dZ = tanh_backward(dA, activation_cache)
        #print("DZrelu is ", dZ)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        #print(dA_prev, dW, db, " are the next layer results ")
    return dA_prev, dW, db

def compute_dAL(AL, Y):
  return 2*(AL - Y)

def L_model_backward(AL, Y, caches):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    dAL = compute_dAL(AL, Y)
    dA_prev, dW, db = linear_activation_backward(dAL, caches[L-1], "linear")
    grads["dA" + str(L-1)] = dA_prev
    grads["dW" + str(L)] = dW
    grads["db" + str(L)] = db
    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        dA_prev, dW, db = linear_activation_backward(dA_prev, caches[l], "relu")
        grads["dA" + str(l)] = dA_prev
        grads["dW" + str(l+1)] = dW
        grads["db" + str(l+1)] = db
    #print(grads, " are the final grads")
    return grads

def update_parameters(params, grads, learning_rate, layers_dims=[]):
    parameters = params.copy()
    L = len(layers_dims) - 1
    for l in range(L):
      parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW"+str(l+1)]
      parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db"+str(l+1)]
    return parameters

def initialize_adam(parameters, layers_dims=[]) :
    L = len(layers_dims)
    v = {}
    s = {}
    
    for l in range(1, L):
      v["dW" + str(l)] = np.zeros(np.shape(parameters["W" + str(l)]))
      v["db" + str(l)] = np.zeros(np.shape(parameters["b" + str(l)]))
      s["dW" + str(l)] = np.zeros(np.shape(parameters["W" + str(l)]))
      s["db" + str(l)] = np.zeros(np.shape(parameters["b" + str(l)]))
    
    return v, s
  
def update_parameters_with_adam(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8,
                                layers_dims=[]):
    L = len(layers_dims)                 # number of layers in the neural networks
    v_corrected = {}                     
    s_corrected = {}
    #print(v, s)
    for l in range(1, L):
      #print("oKKKK", l)
      #print(v['dW1'])
      v["dW" + str(l)] = beta1*v["dW" + str(l)] + (1-beta1)*grads["dW"+ str(l)]
      v["db" + str(l)] = beta1*v["db" + str(l)] + (1-beta1)*grads["db"+ str(l)]
      #print("oKKKK1")
      v_corrected["dW" + str(l)] = v["dW" + str(l)]/(1-np.power(beta1,t))
      v_corrected["db" + str(l)] = v["db" + str(l)]/(1-np.power(beta1,t))
      s["dW" + str(l)] = beta2*s["dW" + str(l)] + (1-beta2)*grads["dW"+ str(l)]*grads["dW"+ str(l)]
      s["db" + str(l)] = beta2*s["db" + str(l)] + (1-beta2)*grads["db"+ str(l)]*grads["db"+ str(l)]
      #print("oKKKK22")
      s_corrected["dW" + str(l)] = s["dW" + str(l)]/(1-np.power(beta2,t))
      s_corrected["db" + str(l)] = s["db" + str(l)]/(1-np.power(beta2,t))

      parameters["W" + str(l)] = parameters["W" + str(l)]- learning_rate*v_corrected["dW" + str(l)]/(np.sqrt(s_corrected["dW" + str(l)])+epsilon)
      parameters["b" + str(l)] = parameters["b" + str(l)]- learning_rate*v_corrected["db" + str(l)]/(np.sqrt(s_corrected["db" + str(l)])+epsilon)

    return parameters

def L_layer_model(X, Y, layers_dims, learning_rate = 0.01, num_iterations = 3000, print_cost=False, dropout=False, dropout_layers = []):
    np.random.seed(1)
    costs = []
    parameters = initialize_parameters_he(layers_dims)
    v,s = initialize_adam(parameters)
    #print(v, s)
    for i in range(0, num_iterations):
      #print("ITERATION NUMBER : ", i)
      AL, caches = L_model_forward(X, parameters, dropout, dropout_layers, layers_dims)
      #print("AL is ", AL)
      cost = compute_cost(AL, Y)
      #print("Cost after ", i, " iteration is ", cost)
      if(cost < 0.1):
        print("Reached convergence, breaking")
        return parameters, costs, AL
      grads = L_model_backward(AL, Y, caches)
      #print("Grads are the ", grads)
      #print(" grads is ", grads)
      parameters = update_parameters_with_adam(parameters, grads, v, s, i+1, layers_dims)
      if print_cost and i % 100 == 0 or i == num_iterations - 1:
        print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        #print("Pramateres on iteration i is ", parameters)
      if i % 10 == 0 or i == num_iterations:
          costs.append(cost)
    
    return parameters, costs, AL

def normalize(data):
    for i in range(0,data.shape[1]):
        data[:,i] = (data[:,i] - np.mean(data[:,i]))/(np.std(data[:, i]))
    return data

def predict_model_v_partial(agent_loc, pred_loc, prey_probs, agent_pred_dist, agent_prey_dist):
  X = []
  prey_prob_list = list(prey_probs.values())
  #print("Prey prob list is ", prey_prob_list)
  X = [[agent_loc] +[pred_loc] + prey_prob_list + [agent_pred_dist] + [agent_prey_dist]]
  X = np.array(X)
  X = X.reshape(X.shape[0], -1).T
  with open('parameters_v_partial_star.pickle', 'rb') as f:
    parameters = pickle.load(f)
  with open('layers_v_partial_star.pickle', 'rb') as f:
    layers_dims = pickle.load(f)
  #print("parameters and layers dims is ", layers_dims)
  predictions = predict_model(X, parameters, layers_dims)
  return predictions[0][0]

def predict_model_v_bonus_partial(agent_loc, pred_loc, prey_probs, agent_pred_dist, agent_prey_dist):
  X = []
  prey_prob_list = list(prey_probs.values())
  #print("Prey prob list is ", prey_prob_list)
  X = [[agent_loc] +[pred_loc] + prey_prob_list + [agent_pred_dist] + [agent_prey_dist]]
  X = np.array(X)
  X = X.reshape(X.shape[0], -1).T
  with open('parameters_v_partial_bonus.pickle', 'rb') as f:
    parameters = pickle.load(f)
  with open('layers_v_partial_bonus.pickle', 'rb') as f:
    layers_dims = pickle.load(f)
  #print("parameters and layers dims is ", layers_dims)
  predictions = predict_model(X, parameters, layers_dims)
  return predictions[0][0]

def predict_model(X, parameters, layers_dims):
  predictions, caches = L_model_forward(X, parameters, layers_dims = layers_dims)
  return predictions

def train_model_v(X, y, layers_dims):
  parameters, costs, prediction = L_layer_model(X, y, layers_dims, num_iterations = 20000, learning_rate = 0.01, print_cost = True)

  with open('parameters_v_star.pickle', 'wb') as handle:
    pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('layers_v_star.pickle', 'wb') as handle:
    pickle.dump(layers_dims, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_model_v_partial(X, y, layers_dims):
  parameters, costs, prediction = L_layer_model(X, y, layers_dims, num_iterations = 20000, learning_rate = 0.01, print_cost = True, dropout=True, dropout_layers=[0, 0.3, 0.4, 0.4, 0.3, 0.3, 0.2, 0.2, 0.2, 0.2, 0])

  with open('parameters_v_partial.pickle', 'wb') as handle:
    pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('layers_v_partial.pickle', 'wb') as handle:
    pickle.dump(layers_dims, handle, protocol=pickle.HIGHEST_PROTOCOL)

def train_model_v_partial_bonus(X, y, layers_dims):
  parameters, costs, prediction = L_layer_model(X, y, layers_dims, num_iterations = 20000, learning_rate = 0.01, print_cost = True, dropout=True, dropout_layers=[0, 0.3, 0.4, 0.3, 0.2, 0.2, 0.2, 0.2, 0])

  with open('parameters_v_partial_bonus.pickle', 'wb') as handle:
    pickle.dump(parameters, handle, protocol=pickle.HIGHEST_PROTOCOL)
  with open('layers_v_partial_bonus.pickle', 'wb') as handle:
    pickle.dump(layers_dims, handle, protocol=pickle.HIGHEST_PROTOCOL)
  return costs

def data_clean_model_v_train(data):
    data.drop(columns={'Reward', 'Policy_prey_dist', 'Policy_pred_dist'}, inplace=True)
    data['Agent_prey_cube'] = data['Agent_Prey_dist']**3
    data['Agent_pred_cube'] = data['Agent_Pred_dist']**3
    data['Agent_loc_cube'] = data['Agent_loc']**3
    data['Prey_loc_cube'] = data['Prey_loc']**3
    data['Pred_loc_cube'] = data['Pred_loc']**3
    df = data.copy(deep=True)
    inf = math.inf
    df = data
    infinities = []
    zeros =[]
    for ind in data.index:
      if(df['Utility'][ind] == inf):
        infinities.append(ind)
      if(df['Utility'][ind] == 0):
        zeros.append(ind)
    drops = infinities + zeros
    df = df.drop(labels = drops, axis=0)
    y = df['Utility'].to_numpy()
    print(y)
    features = df.copy(deep=True)
    features.drop(columns={'Utility'}, inplace=True)
    X = features.to_numpy()
    y = y.reshape(y.shape[0], 1)
    normalize_x = normalize(X)
    y = y.reshape(y.shape[0], -1).T
    X = X.reshape(X.shape[0], -1).T
    return X, y

def data_clean_model_v_partial_train(data):
    df = data.copy(deep=True)
    inf = math.inf
    df = data
    infinities = []
    zeros =[]
    for ind in data.index:
      if(df['Utility'][ind] == inf):
        infinities.append(ind)
      if(df['Utility'][ind] == 0):
        zeros.append(ind)
    drops = infinities + zeros
    df = df.drop(labels = drops, axis=0)
    y = df['Utility'].to_numpy()
    print(y)
    features = df.copy(deep=True)
    features.drop(columns={'Utility'}, inplace=True)
    X = features.to_numpy()
    y = y.reshape(y.shape[0], 1)
    print("X shape is ", X.shape)
    X_new = []
    for i in range(X.shape[0]):
      X_new.append([X[i][0]] + [X[i][1]] + [X[i][2]] + [X[i][3]] + ast.literal_eval(X[i][4]))
    X_arr = np.array(X_new)
    X= X_arr
    normalize_x = normalize(X)
    y = y.reshape(y.shape[0], -1).T
    X = X.reshape(X.shape[0], -1).T
    return X, y

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--v_star', action='store_true', default=False)
  parser.add_argument('--v_partial', action='store_true', default=False)
  parser.add_argument('--train',action='store_true', default=False)
  parser.add_argument('--predict',action='store_true', default=False)
  args = parser.parse_args()
  if args.v_star and args.train:
    with open('layers_v_star.pickle', 'rb') as f:
      layers_dims = pickle.load(f)
    data = pd.read_csv('./u_star_data.csv', index_col=0)
    X, Y = data_clean_model_v_train(data)
    train_model_v(X, Y, layers_dims)
  if args.v_star and args.train:
    with open('layers_v_star.pickle', 'rb') as f:
      layers_dims = pickle.load(f)
    data = pd.read_csv('./u_partial_data.csv', index_col=0)
    X, Y = data_clean_model_v_partial_train(data)
    train_model_v_partial(X, Y, layers_dims)