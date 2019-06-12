from pyspark import SparkContext
import numpy as np

sc = SparkContext("local[*]", "Simple App")


def mysplit(_lines):
    ans = []
    for line in _lines:
        ans.append(line.split(","))
    return ans


def division_to_train_and_test(_lines):
    train__lines = []
    test__lines = []
    train_size = int(len(_lines)*(75/100))
    for i in range(0, train_size):
        train__lines.append([float(j) for j in _lines.__getitem__(i)])
    for j in range(train_size, len(_lines)):
        test__lines.append([float(k) for k in _lines.__getitem__(j)])
    return train__lines, test__lines


def division_to_data_x_and_data_y(_train_lines):
    _data_x = []
    _data_y = []
    for i in range(len(_train_lines)):
        new_item = _train_lines.__getitem__(i)
        new_item = new_item.__getitem__(len(new_item)-2)
        _data_y.append(new_item)
    for i in range(len(_train_lines)):
        new_item = _train_lines.__getitem__(i)
        new_item.remove(new_item.__getitem__(len(new_item)-2))
        _data_x.append(new_item)
    return _data_x, _data_y


def h(x, _w, _b):
    return 1 / (1+np.exp(-(np.dot(x, _w) + _b)))


def check_tests(_test_lines, _w, _b, _predictions):
    _data_x, _data_y = division_to_data_x_and_data_y(_test_lines)
    for i in range(len(_data_x)):
        print("Prediction = ", h(_data_x.__getitem__(i), _w, _b))
        if h(_data_x.__getitem__(i), _w, _b) > 0.5:
            _predictions.append(1.0)
            print("1.0")
        else:
            _predictions.append(0.0)
            print("0.0")
        print("Actual = ", _data_y.__getitem__(i))
    return _predictions, _data_x, _data_y


def accuracy(_predictions, _test_lines, _data_x, _data_y):
    print(_data_y)
    print(_predictions)
    true_positive = 0
    true_negative = 0
    for i in range(len(_predictions)):
        if _data_y.__getitem__(i) == 1 and _predictions.__getitem__(i) == 1:
            true_positive += 1
        if _data_y.__getitem__(i) == 0 and _predictions.__getitem__(i) == 0:
            true_negative += 1
    print("Accuracy = ", (true_positive + true_negative)/len(_predictions))


def recall(_predictions, _test_lines, _data_x, _data_y):
    true_positive = 0
    false_negative = 0
    for i in range(len(_predictions)):
        if _data_y.__getitem__(i) == 1 and _predictions.__getitem__(i) == 1:
            true_positive += 1
        if _data_y.__getitem__(i) == 1 and _predictions.__getitem__(i) == 0:
            false_negative += 1
    ans = true_positive/(true_positive + false_negative)
    print("Recall = ", ans)
    return ans


def precision(_predictions, _test_lines, _data_x, _data_y):
    true_positive = 0
    false_positive = 0
    for i in range(len(_predictions)):
        if _data_y.__getitem__(i) == 1 and _predictions.__getitem__(i) == 1:
            true_positive += 1
        if _data_y.__getitem__(i) == 0 and _predictions.__getitem__(i) == 1:
            false_positive += 1
    ans = true_positive / (true_positive + false_positive)
    print("Precision = ", ans)
    return ans


def fmeasure(_p, _r):
    print("F-measure = ", 2*((_p*_r)/(_p+_r)))
    return 2*((_p*_r)/(_p+_r))


text_file = sc.textFile("prices.txt")
lines = mysplit(text_file.collect())


train_lines, test_lines = division_to_train_and_test(lines)
data_x, data_y = division_to_data_x_and_data_y(train_lines)

data_x = np.array(data_x)
data_y = np.array(data_y)
w = [0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
b = 0
alpha = 0.001
for iteration in range(100000):
    deriv_b = np.mean(1*(h(data_x, w, b) - data_y))
    deriv_w = np.dot((h(data_x, w, b) - data_y), data_x)*1/len(data_y)
    b -= alpha*deriv_b
    w -= alpha*deriv_w

predictions = []
# Q3
predictions, _data_x, _data_y = check_tests(test_lines, w, b, predictions)
# Q4
accuracy(predictions, test_lines, _data_x, _data_y)
r = recall(predictions, test_lines, _data_x, _data_y)
p = precision(predictions, test_lines, _data_x, _data_y)
fmeasure(p, r)
