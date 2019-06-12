
import mnist
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import expit
import matplotlib.cm as cm

def one_hot_encoding(X_train, Y_train, X_test, Y_test):
    X_test = X_test/ 255
    X_train = X_train/ 255
    digits = 10

    length = Y_train.shape[0]
    Z_train = np.zeros((length, digits))
    Z_train[np.arange(length), Y_train] = 1
    Y_train = Z_train

    length = Y_test.shape[0]
    Z_test = np.zeros((length, digits))
    Z_test[np.arange(length), Y_test] = 1
    Y_test = Z_test

    return X_train, Y_train, X_test, Y_test


def shuffle(X_train, Y_train):
    index = np.random.permutation(Y_train.shape[0])
    X_train, Y_train = X_train[index, :], Y_train[index,:]
    return X_train, Y_train

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

    train_size = int(dataset_size * (1 - val_percentage))
    idx_train = idx[:train_size]
    idx_val = idx[train_size:]
    X_train, Y_train = X[idx_train], Y[idx_train]
    X_val, Y_val = X[idx_val], Y[idx_val]
    return X_train, Y_train, X_val, Y_val


def softmax_loss(targets, outputs, weights, lamda):
    targets = np.reshape(targets,outputs.shape)
    assert targets.shape == outputs.shape
    softmax_error = np.multiply(targets, np.log((outputs)))
    mean_softmax = -softmax_error.sum() # med eller uten minus
    regularization = np.sum(np.square(weights))*lamda

    softmaxx_error = (mean_softmax + regularization)/(targets.shape[0])
    return softmaxx_error

def forward_pass(X, w):
    return X.dot(w)


def softmax(z):
    s = np.divide(np.exp(z), np.sum(np.exp(z), axis=1, keepdims=True))
    return s

def gradient_descent(X, outputs, targets, weights, learning_rate, lamda):
    N = X.shape[0]
    targets = np.reshape(targets,outputs.shape)
    assert outputs.shape == targets.shape


    for i in range(weights.shape[0]):
        # Gradient for logistic regression
        dw_i = -(targets-softmax(outputs))*X[:, i:i+1]
        dw_i += 2*lamda*np.sum(weights[i])
        dw_i = dw_i.sum(axis=0)

        weights[i] = weights[i] - (learning_rate * dw_i)/(targets.shape[0])

    return weights

def prediction(X, w):
    outs = forward_pass(X,w)
    outputs = softmax(outs)
    pred = np.argmax(outputs, axis=1) #(outputs > .5)[:, 0 ]

    return pred

def label(Y):
    return np.argmax(Y, axis=1)

def average_labels(X_train, Y_train):
    labels = np.zeros((10, 785))
    Y_train = np.argmax(Y_train, axis=1)
    for i in range(10):
        counter = 0
        for j in range(len(Y_train)):
            if i==Y_train[j]:
                labels[i] += X_train[j]
                counter+= 1
        labels[i] = labels[i]/counter
    return labels



# Hyperparameters
epochs = 40
batch_size = 32

# Tracking variables
TRAIN_LOSS = []
VAL_LOSS = []
TEST_LOSS = []
TRAINING_STEP = []
TRAIN_ACC = []
VAL_ACC = []
TEST_ACC = []

def train_loop(X_train, Y_train, X_val, Y_val, X_test, Y_test):

    num_features = X_train.shape[1]
    num_batches_per_epoch = X_train.shape[0] // batch_size
    check_step = num_batches_per_epoch // 10
    w = np.random.normal(size=(num_features, 10)) * 0.01

    regularization = 1
    lamda = 0.001
    training_it = 0
    T = 1000
    for epoch in range(epochs):
        print(epoch / epochs)


        # shuffle(X_train, Y_train)
        for i in range(num_batches_per_epoch):
            init_learning_rate = 0.1
            learning_rate = init_learning_rate / (1 + training_it/T)
            #learning_rate = 0.0001
            training_it += 1
            X_batch = X_train[i * batch_size:(i + 1) * batch_size]
            Y_batch = Y_train[i * batch_size:(i + 1) * batch_size]

            out = forward_pass(X_batch, w)


            w = gradient_descent(X_batch, out, Y_batch, w, learning_rate, lamda)

            if True: #i % check_step == 0:
                # Training set

                train_out = softmax(forward_pass(X_train, w))
                train_loss = softmax_loss(Y_train, train_out, w, lamda)
                TRAIN_LOSS.append(train_loss)
                TRAINING_STEP.append(training_it)

                val_out = softmax(forward_pass(X_val,w)) # 1/(1+np.exp(-forward_pass(X_val, w)))
                val_loss = softmax_loss(Y_val, val_out, w, lamda)
                VAL_LOSS.append(val_loss)

                test_out = softmax(forward_pass(X_test,w)) # 1/(1+np.exp(-forward_pass(X_test, w)))
                test_loss = softmax_loss(Y_test, test_out, w, lamda)
                TEST_LOSS.append(test_loss)

                TRAIN_ACC.append(100 * np.sum(prediction(X_train, w) == label(Y_train)) / len(Y_train))
                VAL_ACC.append(100 * np.sum(prediction(X_val, w) == label(Y_val)) / len(Y_val))
                TEST_ACC.append(100 * np.sum(prediction(X_test, w) == label(Y_test)) / len(Y_test))




        if (epoch % 1 == 0):
            print("Epoch: %d, Loss: %.8f, Error: %.8f, Val_Loss: %.8f, Val_Error: %.8f "
            % (epoch, train_loss, np.mean(TRAIN_LOSS), val_loss, np.mean(VAL_LOSS)))

    return w


## MAIN

def main():
    #mnist.init()
    X_train, Y_train, X_test, Y_test = mnist.load()
    X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
    X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
    X_train, Y_train, X_test, Y_test = one_hot_encoding(X_train, Y_train, X_test, Y_test)
    #X_train, Y_train = shuffle(X_train, Y_train)

    X_train = X_train[:10000]
    Y_train = Y_train[:10000]
    X_test = X_test[:1000]
    Y_test = Y_test[:1000]

    X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)
    print("Train shape: X: {}, Y: {}".format(X_train.shape, Y_train.shape))
    print("Validation shape: X: {}, Y: {}".format(X_val.shape, Y_val.shape))

    ## TRAINING

    w = train_loop(X_train, Y_train, X_val, Y_val, X_test, Y_test)

    plt.figure(figsize=(12, 8))
    # plt.ylim([0, 1])
    plt.xlabel("Training steps")
    plt.ylabel("Softmax Loss")
    plt.plot(TRAINING_STEP, TRAIN_LOSS, label="Training loss")
    plt.plot(TRAINING_STEP, VAL_LOSS, label="Validation loss")
    plt.plot(TRAINING_STEP, TEST_LOSS, label="Test loss")
    plt.legend()  # Shows graph labels
    plt.show()

    plt.figure(figsize=(12, 8 ))
    #plt.ylim([0, 1])
    plt.xlabel("Training steps")
    plt.ylabel("Classified Correctly")
    plt.plot(TRAINING_STEP, TRAIN_ACC, label="Training accuracy")
    plt.plot(TRAINING_STEP, VAL_ACC, label="Validation accuracy")
    plt.plot(TRAINING_STEP, TEST_ACC, label="Test accuracy")
    plt.legend() # Shows graph labels
    plt.show()

    labels= average_labels(X_train, Y_train)
    nr=4
    nc=5
    fig, axis = plt.subplots(nr, nc)
    images = []
    idx=0
    idx_2 = 0

    for i in range(nr):
        for j in range(nc):
            if (i==0 or i==2):
                images.append(axis[i,j].imshow(labels[idx_2,:-1].reshape(28,28), cmap=plt.get_cmap('seismic')))
                idx_2 +=1
                axis[i,j].label_outer()
            else:
                images.append(axis[i,j].imshow(w[:-1,idx].reshape(28,28), cmap=plt.get_cmap('seismic')))
                idx += 1
                axis[i,j].label_outer()

    plt.show()




if __name__ == "__main__":
    main()
