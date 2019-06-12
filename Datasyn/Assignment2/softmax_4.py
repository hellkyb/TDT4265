import numpy as np
import matplotlib.pyplot as plt
import mnist
#import tqdm
#mnist.init()

def should_early_stop(validation_loss, num_steps=3):
    if len(validation_loss) < num_steps+1:
        return False

    is_increasing = [validation_loss[i] <= validation_loss[i+1] for i in range(-num_steps-1, -1)]
    return sum(is_increasing) == len(is_increasing)

def train_val_split(X, Y, val_percentage):
  """
    Selects samples from the dataset randomly to be in the validation set. Also, shuffles the train set.
    --
    X: [N, num_features] numpy vector,
    Y: [N, 1] numpy vector
    val_percentage: amount of data to put in validation set
  """
  dataset_size = X.shape[0]
  idx = np.arange(0, dataset_size)
  np.random.shuffle(idx)

  train_size = int(dataset_size*(1-val_percentage))
  idx_train = idx[:train_size]
  idx_val = idx[train_size:]
  X_train, Y_train = X[idx_train], Y[idx_train]
  X_val, Y_val = X[idx_val], Y[idx_val]
  return X_train, Y_train, X_val, Y_val

def onehot_encode(Y, n_classes=10):
    onehot = np.zeros((Y.shape[0], n_classes))
    onehot[np.arange(0, Y.shape[0]), Y] = 1
    return onehot

def bias_trick(X):
    return np.concatenate((X, np.ones((len(X), 1))), axis=1)

def check_gradient(X, targets, w, epsilon, computed_gradient, w2):
    print("Checking gradient...")
    dw = np.zeros_like(w)
    for k in range(w.shape[0]):
        for j in range(w.shape[1]):
            new_weight1, new_weight2 = np.copy(w), np.copy(w)
            new_weight1[k,j] += epsilon
            new_weight2[k,j] -= epsilon
            loss1 = cross_entropy_loss(X, targets, [new_weight1, w2])
            loss2 = cross_entropy_loss(X, targets, [new_weight2, w2])
            dw[k,j] = (loss1 - loss2) / (2*epsilon)
    maximum_abosulte_difference = abs(computed_gradient-dw).max()
    assert maximum_abosulte_difference <= epsilon**2, "Absolute error was: {}".format(maximum_abosulte_difference)
    print("Hello")

def softmax(a):
    a_exp = np.exp(a)
    return a_exp / a_exp.sum(axis=1, keepdims=True)

def sigmoid(a):
    return np.divide(1, (1 + np.exp(-a)))

def tanh(a):
    return np.tanh(a)

def tanh_der(a):
    return 1- a**2


def forward(X, w, activation):
    a = X.dot(w.T)
    if activation:
        return softmax(a)
    elif tan:
        return tanh(a)
    else:
        return sigmoid(a)

def calculate_accuracy(X, targets, w):
    #output = forward(X, w)
    y_1 = forward(X, w[0], 0)  # 54000,64
    y_2 = forward(y_1, w[1], 0)  # 54000,10
    y_3 = forward(y_2, w[2], 1)

    predictions = y_3.argmax(axis=1)
    targets = targets.argmax(axis=1)
    return (predictions == targets).mean()

def cross_entropy_loss(X, targets, w):
    y_1 = forward(X, w[0], 0) #54000,64
    y_2 = forward(y_1, w[1], 0) #54000,10
    y_3 = forward(y_2, w[2], 1)

    assert y_3.shape == targets.shape
    #output[output == 0] = 1e-8
    log_y = np.log(y_3)
    cross_entropy = -targets * log_y
    #print(cross_entropy.shape)
    return cross_entropy.mean()

def gradient_descent(X, targets, w, learning_rate, should_check_gradient):
    normalization_factor = X.shape[0] * targets.shape[1] # batch_size * num_classes
    y_1 = forward(X, w[0], 0) #64,64
    y_2 = forward(y_1, w[1], 0)
    y_3 = forward(y_2, w[2], 1) #64,10

    delta_k = - (targets - y_3) #64,10

    dw_3 = delta_k.T.dot(y_2) #np.matmul(y_1.T, delta_k).T #64,10
    # w_2 er 10, 64
    #dw_2 = delta_k.T.dot(outputs)

    d_output_2 = delta_k.dot(w[2])  # np.matmul(w[1].T, delta_k.T) #64,64
    if tan:
        delta_2 = tanh_der(y_2) * d_output_2
    else:
        delta_2 = y_2 * (1 - y_2) * d_output_2  # 64,64

    dw_2 = delta_2.T.dot(y_1)

    d_output_1 = delta_2.dot(w[1]) #np.matmul(w[1].T, delta_k.T) #64,64
    if tan:
        delta_1 = tanh_der(y_1) * d_output_1
    else:
        delta_1 =  y_1*(1 - y_1) * d_output_1 #64,64

    dw_1 = delta_1.T.dot(X) #64,785
    # w_1 er 64,785

    dw_3 = dw_3 / normalization_factor
    dw_2 = dw_2 / normalization_factor # Normalize gradient equally as loss normalization
    dw_1 = dw_1 / normalization_factor
    assert dw_1.shape == w[0].shape, "dw shape was: {}. Expected: {}".format(dw_2.shape, w[0].shape)

    dw = [dw_1, dw_2, dw_3]

    if should_check_gradient:
        check_gradient(X, targets, w[0], 1e-2,  dw[0], w[1])
    w[0] = w[0] - learning_rate * dw[0]
    w[1] = w[1] - learning_rate * dw[1]
    w[2] = w[2] - learning_rate * dw[2]

    return w

def weight_initialization(input_units, output_units, init):
    weight_shape = (output_units, input_units)
    if init:
        return np.random.uniform(-1, 1, weight_shape)
    else:
        return np.random.normal(0, np.divide(1, np.sqrt(input_units)), weight_shape)


X_train, Y_train, X_test, Y_test = mnist.load()

# Pre-process data
X_train, X_test = X_train / 127.5, X_test / 127.5
X_train, X_test = X_train-1, X_test-1
X_train = bias_trick(X_train)
X_test = bias_trick(X_test)
Y_train, Y_test = onehot_encode(Y_train), onehot_encode(Y_test)

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)

# Hyperparameters

batch_size = 128
learning_rate = 0.8
num_batches = X_train.shape[0] // batch_size
should_gradient_check = False
check_step = num_batches // 10
max_epochs = 20
hidden_units = 60
tan = 1

# Tracking variables
TRAIN_LOSS = []
TEST_LOSS = []
VAL_LOSS = []
TRAIN_ACC = []
TEST_ACC = []
VAL_ACC = []
def train_loop():
    global X_train, Y_train
    w1 = weight_initialization(X_train.shape[1],hidden_units, 0)
       # np.random.rand(hidden_units, X_train.shape[1])
    w2 = weight_initialization(hidden_units, hidden_units, 0)
        #np.random.rand(Y_train.shape[1], hidden_units)
    w3 = weight_initialization(hidden_units, Y_train.shape[1], 0)
    w = [w1, w2, w3]

    for e in range(max_epochs): # Epochs
        for i in range(num_batches):
            X_batch = X_train[i*batch_size:(i+1)*batch_size]
            Y_batch = Y_train[i*batch_size:(i+1)*batch_size]

            w = gradient_descent(X_batch, Y_batch, w, learning_rate, should_gradient_check)
            #print(cross_entropy_loss(X_batch, Y_batch, w))
            if i % check_step == 0:
                # Loss
                TRAIN_LOSS.append(cross_entropy_loss(X_train, Y_train, w))
                TEST_LOSS.append(cross_entropy_loss(X_test, Y_test, w))
                VAL_LOSS.append(cross_entropy_loss(X_val, Y_val, w))


                TRAIN_ACC.append(calculate_accuracy(X_train, Y_train, w))
                VAL_ACC.append(calculate_accuracy(X_val, Y_val, w))
                TEST_ACC.append(calculate_accuracy(X_test, Y_test, w))

                if should_early_stop(VAL_LOSS):
                    print(VAL_LOSS[-4:])
                    print("early stopping.")
                    return w[1]

        if (e % 1 == 0):
            print("Epoch: %d, Loss: %.8f, Acc: %.8f, Val_Loss: %.8f, Val_Acc: %.8f "
                 % (e, TRAIN_LOSS[-1], TRAIN_ACC[-1], VAL_LOSS[-1], VAL_ACC[-1] ))

    return w



w = train_loop()

plt.plot(TRAIN_LOSS, label="Training loss")
plt.plot(TEST_LOSS, label="Testing loss")
plt.plot(VAL_LOSS, label="Validation loss")
plt.legend()
#plt.ylim([0, 0.05])
plt.show()

plt.clf()
plt.plot(TRAIN_ACC, label="Training accuracy")
plt.plot(TEST_ACC, label="Testing accuracy")
plt.plot(VAL_ACC, label="Validation accuracy")
#plt.ylim([0.8, 1.0])
plt.legend()
plt.show()

plt.clf()
'''
w = w[:, :-1] # Remove bias
w = w.reshape(10, 28, 28)
w = np.concatenate(w, axis=0)
plt.imshow(w, cmap="gray")
plt.show()
'''
