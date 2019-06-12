
import mnist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.special import expit
#mnist.init()
X_train, Y_train, X_test, Y_test = mnist.load()

X_train = np.concatenate((X_train, np.ones((X_train.shape[0], 1))), axis=1)
X_test = np.concatenate((X_test, np.ones((X_test.shape[0], 1))), axis=1)
print("X shape with bias:", X_train.shape)

def remove_all_but_twos_and_threes(X_train, Y_train, X_test, Y_test):
    count_train = 0
    count_test = 0
    for i in Y_train:
        if (i == 2 or i == 3):
            count_train = count_train + 1
    for i in Y_test:
        if (i == 2 or i == 3):
            count_test = count_test + 1
    X_train_2 = np.zeros([count_train, 785])
    X_test_2 = np.zeros([count_test, 785])
    Y_train_2 = np.zeros(count_train)
    Y_test_2 = np.zeros(count_test)
    count = 0
    for i in range(0,len(Y_train)):
        if (Y_train[i] == 2 or Y_train[i] == 3):
            Y_train_2[count] = Y_train[i]
            X_train_2[count] = X_train[i]
            count = count + 1
    count = 0
    for i in range(0,len(Y_test)):
        if (Y_test[i] == 2 or Y_test[i] == 3):
            Y_test_2[count] = Y_test[i]
            X_test_2[count] = X_test[i]
            count = count + 1

    X_test = (X_test_2)
    X_test /= 255
    Y_test = Y_test_2-2
    X_train = X_train_2
    X_train /= 255
    Y_train = Y_train_2-2
    return X_train, Y_train, X_test, Y_test

X_train, Y_train, X_test, Y_test = remove_all_but_twos_and_threes(X_train,Y_train, X_test, Y_test)
print(X_train.shape)
print(Y_train.shape)

X_train = X_train[:10000]
Y_train = Y_train[:10000]
X_test = X_test[-1000:]
Y_test = Y_test[-1000:]

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

X_train, Y_train, X_val, Y_val = train_val_split(X_train, Y_train, 0.1)
print("Train shape: X: {}, Y: {}".format(X_train.shape, Y_train.shape))
print("Validation shape: X: {}, Y: {}".format(X_val.shape, Y_val.shape))

def logistic_loss(targets, outputs):
    targets = np.reshape(targets,outputs.shape)
    assert targets.shape == outputs.shape
    log_error = targets*np.log(outputs) + (1-targets)*np.log(1-outputs)
    mean_log_error = -log_error.mean()
    return mean_log_error

def logistic_loss_regularization(targets, outputs, weights, lamda):
    targets = np.reshape(targets,outputs.shape)
    assert targets.shape == outputs.shape
    log_error = targets*np.log(outputs) + (1-targets)*np.log(1-outputs)
    mean_log_error = -log_error.sum()
    regularization =  np.sum(np.square(weights))*lamda
    #print(regularization.shape)
    #print(mean_log_error.shape)
    logg_error = (mean_log_error + regularization)/(targets.shape[0])
    return logg_error

def forward_pass(X, w):
    return X.dot(w)

def gradient_descent(X, outputs, targets, weights, learning_rate, regularization, lamda):
    N = X.shape[0]

    targets = np.reshape(targets,outputs.shape)
    assert outputs.shape == targets.shape

    for i in range(weights.shape[0]):
        # Gradient for logistic regression

        dw_i = -(targets-1/(1+np.exp(-outputs)))*X[:, i:i+1]
        if regularization:
            dw_i += 2*lamda*np.sum(weights)
        dw_i = dw_i.sum(axis=0)/(targets.shape[0])

        weights[i] = weights[i] - learning_rate * dw_i

    return weights

def prediction(X, w):
    outs = forward_pass(X,w)
    outputs = np.divide(1, (1+np.exp(-outs)))
    pred = (outputs > .5)[:, 0]
    return pred


## TRAINING

# Hyperparameters
epochs = 40
batch_size = 32

# Tracking variables
TRAIN_LOSS = []
VAL_LOSS = []
TRAINING_STEP = []
TEST_LOSS = []
TRAIN_ACC = []
VAL_ACC = []
TEST_ACC = []
WEIGHTS = []

'''
VAL_LOSS_1 = []
VAL_LOSS_2 = []
VAL_LOSS_3 = []
'''
num_features = X_train.shape[1]

num_batches_per_epoch = X_train.shape[0] // batch_size
print(num_batches_per_epoch)
check_step = num_batches_per_epoch // 10

def early_stopping(index):
    if ((VAL_LOSS[index] > VAL_LOSS[index-num_batches_per_epoch-1]) and (VAL_LOSS[index] > VAL_LOSS[index-2-num_batches_per_epoch]) and (VAL_LOSS[index] > VAL_LOSS[index-3-num_batches_per_epoch])):
        print("Yep")
        return 1
    else:
        return 0

def train_loop(lamda):

    np.random.seed(0)
    w = np.random.normal(size=(num_features, 1 ))*0.01

    regularization = 1
    training_it = 0
    T = 1000
    for epoch in range(epochs):
        print(epoch / epochs)
        # shuffle(X_train, Y_train)
        for i in range(num_batches_per_epoch):
            init_learning_rate = 0.01
            learning_rate = init_learning_rate / (1 + training_it/T)
            #print(learning_rate)
            #learning_rate = 0.0001
            X_batch = X_train[i * batch_size:(i + 1) * batch_size]
            Y_batch = Y_train[i * batch_size:(i + 1) * batch_size]

            out = forward_pass(X_batch, w)
            w = gradient_descent(X_batch, out, Y_batch, w, learning_rate, regularization, lamda)
            training_it += 1

            if True: #i % check_step == 0:
                # Training set
                train_out = forward_pass(X_train, w)
                train_out = np.divide(1,(1+np.exp(-train_out)))
                if regularization:
                    train_loss = logistic_loss_regularization(Y_train, train_out, w, lamda)
                else:
                    train_loss = logistic_loss(Y_train, train_out)
                TRAIN_LOSS.append(train_loss)
                TRAINING_STEP.append(training_it)

                val_out = 1/(1+np.exp(-forward_pass(X_val, w)))
                if regularization:
                    val_loss = logistic_loss_regularization(Y_val, val_out, w, lamda)
                else:
                    val_loss = logistic_loss(Y_val, val_out)
                VAL_LOSS.append(val_loss)

                test_out = 1 / (1 + np.exp(-forward_pass(X_test, w)))
                if regularization:
                    test_loss = logistic_loss_regularization(Y_test, test_out, w, lamda)
                else:
                    test_loss = logistic_loss(Y_test, test_out)
                TEST_LOSS.append(test_loss)

                TRAIN_ACC.append(100 * np.sum(prediction(X_train, w) == Y_train) / len(Y_train))
                VAL_ACC.append(100 * np.sum(prediction(X_val, w) == Y_val) / len(Y_val))
                TEST_ACC.append(100 * np.sum(prediction(X_test, w) == Y_test) / len(Y_test))
                WEIGHTS.append(w)



        if epoch > 5:
            if early_stopping(len(VAL_LOSS)-1):
               break


        if (epoch % 1 == 0):
            print("Epoch: %d, Loss: %.8f, Error: %.8f, Val_Loss: %.8f, Val_Error: %.8f "
            % (epoch, train_loss, np.mean(TRAIN_LOSS), val_loss, np.mean(VAL_LOSS)))

    return w

TRAIN_LOSS = []
VAL_LOSS = []
TRAINING_STEP = []
TEST_LOSS = []
TRAIN_ACC = []
VAL_ACC = []
TEST_ACC = []
VAL_ACC = []
WEIGHTS = []
w = train_loop(0.01)
w_1 = w
VAL_ACC_1 = VAL_ACC
WEIGHTS_1 = WEIGHTS
VAL_ACC = []
TRAIN_LOSS = []
VAL_LOSS = []
TRAINING_STEP = []
TEST_LOSS = []
TRAIN_ACC = []
VAL_ACC = []
TEST_ACC = []
WEIGHTS = []
w = train_loop(0.001)
w_2 = w
VAL_ACC_2 = VAL_ACC
WEIGHTS_2 = WEIGHTS
VAL_ACC = []
TRAIN_LOSS = []
VAL_LOSS = []
TRAINING_STEP = []
TEST_LOSS = []
TRAIN_ACC = []
VAL_ACC = []
TEST_ACC = []
WEIGHTS = []
w = train_loop(0.0001)
w_3 = w
VAL_ACC_3 = VAL_ACC
WEIGHTS_3 = WEIGHTS



plt.figure(figsize=(12, 8 ))
#plt.ylim([0, 1])
plt.xlabel("Training steps")
plt.ylabel("Logistic Loss")
plt.plot(TRAINING_STEP, TRAIN_LOSS, label="Training loss")
plt.plot(TRAINING_STEP, VAL_LOSS, label="Validation loss")
plt.plot(TRAINING_STEP, TEST_LOSS, label="Test loss")
plt.legend() # Shows graph labels
plt.show()


plt.figure(figsize=(12, 8 ))
#plt.ylim([0, 1])
plt.xlabel("Training steps")
plt.ylabel("Logistic Loss")
plt.plot(TRAINING_STEP, VAL_ACC_1, label="Lambda = 0.01")
plt.plot(TRAINING_STEP, VAL_ACC_2, label="Lambda = 0.001")
plt.plot(TRAINING_STEP, VAL_ACC_3, label="Lambda = 0.0001")
plt.legend() # Shows graph labels
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

plt.figure(figsize=(12, 8 ))
beep = []
for weights in WEIGHTS_1:
    beep.append(np.linalg.norm(weights))
plt.plot(beep, label="Lambda= 0.001")

beep = []
for weights in WEIGHTS_2:
    beep.append(np.linalg.norm(weights))
plt.plot(beep, label="Lambda= 0.001")
beep = []
for weights in WEIGHTS_3:
    beep.append(np.linalg.norm(weights))
plt.plot(beep,  label="Lambda = 0.0001")
plt.title('Weights with lambda')
plt.show()


ws = []
ws.append(w_1)
ws.append(w_2)
ws.append(w_3)
plt.figure(figsize=(12, 8 ))
plt.subplot(1,3,1)
plt.imshow(ws[0][:-1].reshape(28,28), cmap=plt.get_cmap('seismic'))
plt.subplot(1,3,2)
plt.imshow(ws[1][:-1].reshape(28,28), cmap=plt.get_cmap('seismic'))
plt.subplot(1,3,3)
plt.imshow(ws[2][:-1].reshape(28,28), cmap=plt.get_cmap('seismic'))
plt.show()
