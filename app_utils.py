import numpy as np

def linear_forward(a, w, b):
    z = np.dot(w,a) + b

    assert(z.shape == (w.shape[0], a.shape[1])),"Shape of calculated Z is incorrect in linear_forward function"
    cache = (a, w, b)

    return z, cache

def relu(z):
    a = z * (z > 0)
    activation_cache = {"Z": z}

    assert(a.shape == z.shape),"a's shape is not same as z in relu function"
    return a, activation_cache


def sigmoid(z):
    a = 1/(1 + np.exp(-1*z))
    activation_cache = {"Z": z}

    assert(a.shape == z.shape),"a's shape is not same as z in sigmoid function"
    return a, activation_cache


def linear_activation_forward(a_prev, w, b, activation):
    if activation == "relu":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = relu(z)

    elif activation == "sigmoid":
        z, linear_cache = linear_forward(a_prev, w, b)
        a, activation_cache = sigmoid(z)

    assert (a.shape == (w.shape[0], a_prev.shape[1])),"Calculated shape of A is incorrect"

    cache = (linear_cache, activation_cache)

    return a, cache

def L_model_forward(x, parameters):
    cache_main = []
    a = x
    num_layers = len(parameters) // 2

    for l in range(1, num_layers):
        a_prev = a
        a, cache_temp = linear_activation_forward(a_prev, parameters["W" + str(l)], 
                                                  parameters["b" + str(l)], "relu")
        cache_main.append(cache_temp)

    al, cache_temp = linear_activation_forward(a, parameters["W" + str(num_layers)],
                                               parameters["b" + str(num_layers)], "sigmoid")
    cache_main.append(cache_temp)

    assert (al.shape == (1, x.shape[1]))

    return al, cache_main
