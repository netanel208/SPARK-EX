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
    train_size = int(len(_lines)*(85/100))
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
        new_item = new_item.__getitem__(len(new_item)-1)
        _data_y.append(new_item)
    for i in range(len(_train_lines)):
        new_item = _train_lines.__getitem__(i)
        new_item.remove(new_item.__getitem__(len(new_item)-1))
        _data_x.append(new_item)
    return _data_x, _data_y


def check_tests_and_MSE(_test_lines, _w, _b):
    _data_x, _data_y = division_to_data_x_and_data_y(_test_lines)
    _sum = 0
    for i in range(len(_data_x)):
        y1 = np.dot(_data_x.__getitem__(i), _w) + _b
        print("Prediction = ", y1)
        y2 = _data_y.__getitem__(i)
        print("Actual = ", y2)
        ans = y1 - y2
        _sum += ans**2
    ans = (1/len(_data_x)) * _sum
    print("Mean Squared-Error = ", ans)


text_file = sc.textFile("prices.txt")
lines = mysplit(text_file.collect())


train_lines, test_lines = division_to_train_and_test(lines)
data_x, data_y = division_to_data_x_and_data_y(train_lines)

data_x = np.array(data_x)
data_y = np.array(data_y)
w = [0., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
b = 0
alpha = 0.001
for iteration in range(1000000):
    deriv_b = np.mean(1*((np.dot(data_x, w)+b)-data_y))
    gradient_w = 1.0/len(data_y) * np.dot(((np.dot(data_x, w)+b)-data_y), data_x)
    b -= alpha*deriv_b
    w -= alpha*gradient_w

# Q1+Q2
check_tests_and_MSE(test_lines, w, b)


