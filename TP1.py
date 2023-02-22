import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KernelDensity
import matplotlib.pyplot as plt
from numpy import genfromtxt


def split_data_types(x, y):
    zeros = x[y[:, 0] == 0]
    ones = x[y[:, 0] == 1]
    return zeros, ones


def plotData(x, y, xlabel, ylabel, classifierName):
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="upper left")
    plt.plot(x, y)
    plt.savefig(classifierName)
    plt.show()
    plt.close()


def run_collision_avoidance():
    filename = 'SatelliteConjunctionDataRegression.csv'
    data = genfromtxt(filename, delimiter=',')[1:]
    scaler = StandardScaler()
    scaler.fit(data)
    scaled_data = scaler.transform(data)
    train, test = train_test_split(scaled_data, train_size=0.80)
    train_redone, val = train_test_split(train, train_size=0.80)
    train_x = train_redone[:, :-1]
    train_y = train_redone[:, -1]
    val_x = val[:, :-1]
    val_y = val[:, -1]
    test_x = test[:, :-1]
    test_y = test[:, -1]
    mse_train = []
    mse_val = []
    for degree in range(1, 7):
        poly = PolynomialFeatures(degree)
        transformed = poly.fit_transform(train_x)
        model = LinearRegression().fit(transformed, train_y)
        train_predicted = model.predict(transformed)
        val_predicted = model.predict(poly.transform(val_x))
        mse_train.append(mean_squared_error(train_y, train_predicted))
        mse_val.append(mean_squared_error(val_y, val_predicted))
        plt.figure()
        plt.plot(train_predicted, train_y, color='blue', label='train')
        plt.plot(val_predicted, val_y, color='red', label='val')
        plt.plot(range(-3, 7), range(-3, 7), color='black')
        plt.legend(loc="upper left")
        plt.savefig(f'REGRESS-PRED-VS-TRUE_{degree}')
        plt.close()
    plt.figure()
    plt.plot(range(1, 7), mse_train, color='blue', label='train', marker='s')
    plt.plot(range(1, 7), mse_val, color='red', label='val', marker='x')
    plt.legend(loc="upper left")
    plt.savefig(f'REGRESS-TR-VAL')
    plt.close()
    train_x = train[:, :-1]
    train_y = train[:, -1]
    best_degree = np.argmin(mse_val) + 1
    poly = PolynomialFeatures(best_degree)
    transformed = poly.fit_transform(train_x)
    model = LinearRegression().fit(transformed, train_y)
    test_predicted = model.predict(poly.transform(test_x))
    true_error = mean_absolute_error(test_y, test_predicted)
    print(f'Degree of Best Model : {best_degree}\nPredicted Error of Best Model : {true_error}')


run_collision_avoidance()


def get_divided_set(set):
    zeros = set[set[:, -1] == 0][:, :-1]
    ones = set[set[:, -1] == 1][:, :-1]
    return zeros, ones


def fit_kdes(zeros, ones, bandwidth):
    zeros_kdes, ones_kdes = [], []
    for kde_number in range(0, 4):
        zeros_kdes.append(
            KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(zeros[:, kde_number].reshape(-1, 1)))
        ones_kdes.append(KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(ones[:, kde_number].reshape(-1, 1)))
    return zeros_kdes, ones_kdes


def classify(zeros_log, ones_log, values, zeros_kde, ones_kde):
    classes = np.zeros(values.shape[0])

    for rowNumber in range(values.shape[0]):
        row = values[rowNumber]
        zero_prob = zeros_log
        one_prob = ones_log
        for kde_number in range(0, 4):
            zero_prob += zeros_kde[kde_number].score([[row[kde_number]]])
            one_prob += ones_kde[kde_number].score([[row[kde_number]]])
        if zero_prob < one_prob:
            classes[rowNumber] = 1
    return classes


def get_best_bandwidth(values):
    best_score = 20000
    min_bandwidth = 0
    folds = 5
    kf = StratifiedKFold(folds)
    bandwidths = []
    scores = []
    Xs = values[:, :-1]
    Ys = values[:, -1]

    for curr in np.arange(0.02, 0.6, 0.02):
        curr_score = 0
        for training_ix, validation_ix in kf.split(Xs, Ys):
            train = values[training_ix]
            validation = values[validation_ix]
            train_0, train_1 = get_divided_set(train)
            validation_0, validation_1 = get_divided_set(validation)
            tot_len = train.shape[0]

            zero_log = np.log(float(train_0.shape[0]) / tot_len)
            one_log = np.log(float(train_1.shape[0]) / tot_len)

            zeros_kdes, ones_kdes = fit_kdes(train_0, train_1, curr)

            va_c_zero = classify(zero_log, one_log, validation_0, zeros_kdes, ones_kdes)
            va_c_one = classify(zero_log, one_log, validation_1, zeros_kdes, ones_kdes)

            local_score = (2 - (accuracy_score(np.zeros(va_c_zero.shape), va_c_zero) + \
                                accuracy_score(np.ones(va_c_one.shape), va_c_one))) / 2

            curr_score += local_score
        bandwidths.append(curr)
        curr_avg_score = (curr_score / folds)
        scores.append(curr_avg_score)
        print(f'Bandwidth: {curr}\t Error: {curr_avg_score}')
        if curr_score < best_score:
            best_score = curr_score
            min_bandwidth = curr
    plotData(bandwidths, scores, 'Bandwidths', 'Error', 'NB.png')
    return min_bandwidth


def mcnemar_test(train_0_predicted, train_1_predicted, test_0_predicted, test_1_predicted):
    train_0_size = train_0_predicted.size
    train_1_size = train_1_predicted.size

    test_0_size = test_0_predicted.size
    test_1_size = test_1_predicted.size

    train_0_false = sum(train_0_predicted)
    train_0_true = train_0_size - train_0_false

    train_1_true = sum(train_1_predicted)
    train_1_false = train_1_size - train_1_true

    test_0_false = sum(test_0_predicted)
    test_0_true = test_0_size - test_0_false

    test_1_true = sum(test_1_predicted)
    test_1_false = test_1_size - test_1_true

    training_0_score = 1 - float(sum(train_0_predicted) / train_0_size)
    training_1_score = 1 - float(sum(1 - train_1_predicted) / train_1_size)
    test_0_score = 1 - float(sum(test_0_predicted) / test_0_size)
    test_1_score = 1 - float(sum(1 - test_1_predicted) / test_1_size)
    mcnemar_test_statistic = (pow((test_1_false - test_0_false), 2)) / (test_1_false + test_0_false)
    print("Training 0:", train_0_size, "True 0:", train_0_true, "False 0:", train_0_false, "0 Accuracy: ",
          training_0_score)
    print("Training 1:", train_1_size, "True 1:", train_1_true, "False 1:", train_1_false, "1 Accuracy: ",
          training_1_score)
    print("Test 0:", test_0_size, "True 0:", test_0_true, "False 0:", test_0_false, "0 Accuracy: ", test_0_score)
    print("Test 1:", test_1_size, "True 1:", test_1_true, "False 1:", test_1_false, "1 Accuracy: ", test_1_score)
    print(f'McNemar Test Statistic: {mcnemar_test_statistic}')


def approximate_normal_test(test_0_predicted, test_1_predicted, function_name):
    size = test_0_predicted.size + test_1_predicted.size
    error = sum(test_0_predicted) + (sum(1 - test_1_predicted))

    error_perc = float(error) / size
    standard_deviation = np.sqrt(size * error_perc * (1 - error_perc))
    print(
        f'Confidence interval of the approximate normal test for {function_name}: {error} +- {standard_deviation * 1.96}')


def naive_bayes(best_bandwidth, train_0, train_1, test_0, test_1, tot_len):
    zero_log = np.log(float(train_0.shape[0]) / tot_len)
    one_log = np.log(float(train_1.shape[0]) / tot_len)
    zeros_kdes, ones_kdes = fit_kdes(train_0, train_1, best_bandwidth)
    train_0_predicted = classify(zero_log, one_log, train_0, zeros_kdes, ones_kdes)
    train_1_predicted = classify(zero_log, one_log, train_1, zeros_kdes, ones_kdes)
    test_0_predicted = classify(zero_log, one_log, test_0, zeros_kdes, ones_kdes)
    test_1_predicted = classify(zero_log, one_log, test_1, zeros_kdes, ones_kdes)
    print('-----TESTS FOR NAIVE BAYES------')
    mcnemar_test(train_0_predicted, train_1_predicted, test_0_predicted, test_1_predicted)
    approximate_normal_test(test_0_predicted, test_1_predicted, 'Naive Bayes')
    true_error_estimate = (2 - (
            accuracy_score(np.zeros(test_0_predicted.shape), test_0_predicted) + accuracy_score(
        np.ones(test_1_predicted.shape), test_1_predicted))) / 2
    print(f'True Error Estimate : {true_error_estimate}')


def gaussian(training_x, training_y, train_0, train_1, test_0, test_1):
    gnb = GaussianNB()
    gnb.fit(training_x, training_y)
    train_0_predicted = gnb.predict(train_0)
    train_1_predicted = gnb.predict(train_1)
    test_0_predicted = gnb.predict(test_0)
    test_1_predicted = gnb.predict(test_1)
    print('-----TESTS FOR GAUSSIAN NB------')

    mcnemar_test(train_0_predicted, train_1_predicted, test_0_predicted, test_1_predicted)
    approximate_normal_test(test_0_predicted, test_1_predicted, 'Gaussian NB')
    true_error_estimate = (2 - (
            accuracy_score(np.zeros(test_0_predicted.shape), test_0_predicted) + accuracy_score(
        np.ones(test_1_predicted.shape), test_1_predicted))) / 2
    print(f'True Error Estimate : {true_error_estimate}')


def run_banknotes():
    train = genfromtxt('TP1_train.tsv', delimiter='\t')
    test = genfromtxt('TP1_test.tsv', delimiter='\t')
    training_x, training_y = train[:, :-1], train[:, -1]
    best_bandwidth = get_best_bandwidth(train)
    print(f'Best Bandwidth: {best_bandwidth}')
    train_0, train_1 = get_divided_set(train)
    test_0, test_1 = get_divided_set(test)
    tot_len = train.shape[0]
    naive_bayes(best_bandwidth, train_0, train_1, test_0, test_1, tot_len)
    gaussian(training_x, training_y, train_0, train_1, test_0, test_1)

# run_banknotes()
