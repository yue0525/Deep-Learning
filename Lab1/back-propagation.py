import numpy as np
import matplotlib.pyplot as plt


def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)

    return np.array(inputs), np.array(labels).reshape(n, 1)


def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1 * i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize=18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')

    plt.figure()
    plt.title('Learning curve', fontsize=18)
    plt.plot(learning_array, loss_array)
    plt.show()


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x, 1 - x)


def MSE(y, y_pred):
    mse = np.mean((y - y_pred)**2)
    return mse


def derivative_MSE(y, y_pred):
    return 2 * (y_pred - y)


def forward_pass(X, init_W, hidden1_W, hidden2_W, hidden1_Z, hidden2_Z, hiddenO_Z, a1_Z, a2_Z, aO_Z):
    hidden1_Z = init_W @ X
    a1_Z = sigmoid(hidden1_Z)
    hidden2_Z = hidden1_W @ a1_Z
    a2_Z = sigmoid(hidden2_Z)
    hiddenO_Z = hidden2_W @ a2_Z
    aO_Z = sigmoid(hiddenO_Z)
    # print(aO_Z)
    return init_W, hidden1_W, hidden2_W, hidden1_Z, hidden2_Z, hiddenO_Z, a1_Z, a2_Z, aO_Z


def input_case():
    case = input("generate_linear(0) or generate_XOR_easy(1) : ")
    epoch = 0
    print_time = 0
    if case == "0":
        x, y = generate_linear(n=100)
        epoch = 10000
        print_time = 100
        return x, y, epoch, print_time
    if case == "1":
        x, y = generate_XOR_easy()
        epoch = 100000
        print_time = 1000
        return x, y, epoch, print_time


if __name__ == "__main__":

    learning_rate = 0.01
    init_W = np.random.uniform(0, 1, size=(4, 2))
    hidden1_W = np.random.uniform(0, 1, size=(4, 4))
    hidden2_W = np.random.uniform(0, 1, size=(1, 4))
    hidden1_Z = np.zeros((4, 1))
    hidden2_Z = np.zeros((4, 1))
    hiddenO_Z = np.zeros((1, 1))
    a1_Z = np.zeros((4, 1))
    a2_Z = np.zeros((4, 1))
    aO_Z = np.zeros((1, 1))
    hidden1_dc_dz = np.zeros((4, 1))
    hidden2_dc_dz = np.zeros((4, 1))
    hiddenO_dc_dz = np.zeros((1, 1))
    x, y, epoch, print_time = input_case()
    test_y_pred = []
    learning_array = []
    loss_array = []
    # training
    for i in range(epoch):
        # loss = 0

        for x_num in range(x.shape[0]):
            X = x[x_num].reshape(2, 1)
            # forward_pass
            init_W, hidden1_W, hidden2_W, hidden1_Z, hidden2_Z, hiddenO_Z, a1_Z, a2_Z, aO_Z = forward_pass(
                X, init_W, hidden1_W, hidden2_W, hidden1_Z, hidden2_Z, hiddenO_Z, a1_Z, a2_Z, aO_Z)

            # backward_pass
            derivative_loss = derivative_MSE(y[x_num], aO_Z)
            derivative_sigmoid_hiddenO_Z = derivative_sigmoid(aO_Z)
            hiddenO_dc_dz = derivative_sigmoid_hiddenO_Z * derivative_loss

            hidden2_dc_dz = hidden2_W.T @ hiddenO_dc_dz
            hidden2_dc_dz *= derivative_sigmoid(a2_Z)

            hidden1_dc_dz = hidden1_W.T @ hidden2_dc_dz
            hidden1_dc_dz *= derivative_sigmoid(a1_Z)

            # update
            init_W -= learning_rate * (hidden1_dc_dz @ X.T)
            hidden1_W -= learning_rate * (hidden2_dc_dz @ a1_Z.T)
            hidden2_W -= learning_rate * (hiddenO_dc_dz @ a2_Z.T)

        y_pred = []
        for n in range(x.shape[0]):
            X = x[n].reshape(2, 1)
            init_W, hidden1_W, hidden2_W, hidden1_Z, hidden2_Z, hiddenO_Z, a1_Z, a2_Z, aO_Z = forward_pass(
                X, init_W, hidden1_W, hidden2_W, hidden1_Z, hidden2_Z, hiddenO_Z, a1_Z, a2_Z, aO_Z)
            y_pred.append(aO_Z[0][0])
        y_pred = np.array(y_pred).reshape(y.shape)
        loss = MSE(y, y_pred)
        if i % print_time == 0:
            print(f'epoch {i}: loss : {loss}')
            learning_array.append(i)
            loss_array.append(loss)

        if i == epoch-1:
            test_y_pred = y_pred
    # testing
    test_label = []
    for i in range(test_y_pred.shape[0]):
        if test_y_pred[i] >= 0.5:
            test_label.append(1)
        else:
            test_label.append(0)
    correct_count = 0
    for i in range(len(test_label)):
        if test_label[i] == y[i]:
            correct_count += 1
    print(test_y_pred)
    print(f'accuracy : {(correct_count/len(test_label))*100}%')

    show_result(x, y, test_label)
