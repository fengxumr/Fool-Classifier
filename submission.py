import helper
import math
import numpy as np


def fool_classifier(test_data):
    with open(test_data, 'r') as file:
        data = [line.strip().split(' ') for line in file]    
    
    strategy_instance=helper.strategy() 
    parameters={'gamma': 'auto', 'C': 1.0, 'kernel': 'linear', 'degree': 3, 'coef0': 0.0}
    
    x_train, y_train, word_list = train_generator(strategy_instance.class0, strategy_instance.class1)
    clf = strategy_instance.train_svm(parameters, x_train, y_train)

    data_modified(word_list, clf.coef_[0], data)

    with open('modified_data.txt', 'w') as file:
        file.write('\n'.join([' '.join(a) for a in data]))
    
    modified_data='./modified_data.txt'
    assert strategy_instance.check_data(test_data, modified_data)
    return strategy_instance


def train_generator(class0, class1):
    word_set = set([b for a in class0 for b in a]) | set([b for a in class1 for b in a])
    class_length = len(class0) + len(class1)

    word_count_dict = dict.fromkeys(word_set, 0)
    for line in (class0 + class1):
        for word in set(line):
            word_count_dict[word] += 1

    x_train = []
    y_train = []

    for line in class0:
        word_dict = dict.fromkeys(word_set, 0)
        for word in line:
            word_dict[word] += 1
        line_length = len(line)
        for word in word_dict:
            word_dict[word] = (word_dict[word] / line_length) * math.log(class_length / word_count_dict[word])
        x_train.append([a[1] for a in sorted(list(word_dict.items()))])
        y_train.append(0)

    for line in class1:
        word_dict = dict.fromkeys(word_set, 0)
        for word in line:
            word_dict[word] += 1
        line_length = len(line)
        for word in word_dict:
            word_dict[word] = (word_dict[word] / line_length) * math.log(class_length / word_count_dict[word])
        x_train.append([a[1] for a in sorted(list(word_dict.items()))])
        y_train.append(1)

    return np.array(x_train), np.array(y_train), sorted(list(word_set))


def data_modified(word_list, priority, data):
    assert(len(word_list) == len(priority))
    word_len = len(word_list)
    word_order = sorted([[priority[i], word_list[i]] for i in range(len(word_list))])
    word0_order = [a[1] for a in word_order if a[0] <= 0]
    word1_order = [a[1] for a in word_order if a[0] > 0][::-1]
    
    for line in data:
        count = 20
        for word1 in word1_order:
            if word1 in line:
                count -= 1
            while word1 in line:
                line.pop(line.index(word1))
            if count == 0:
                break
        idx = 0
        while count:
            while word0_order[idx] in line:
                idx += 1
            line.append(word0_order[idx])
            count -= 1











