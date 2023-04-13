import pandas as pd
import numpy as np
import math

TRAIN_FILE = 'heart-train.csv'
TEST_FILE = 'heart-test.csv'

N_STEPS = 10
STEP_SIZE = 0.0001

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
def netflix_or_heart():
    if (TRAIN_FILE == 'netflix-train.csv' or TRAIN_FILE == 'heart-train.csv'):
        return True
    return False
def initialize_list(num_lines):
    return [0] * num_lines

def training():
    df = pd.read_csv(TRAIN_FILE)
    cols = len(df.axes[1]) 
    num_features = cols - 1
    print(num_features)
    count_y_0 = 0
    count_y_1 = 0
    num_lines = 0 
    with open(TRAIN_FILE) as file:
        lines = file.readlines()[1:]
        if netflix_or_heart():
            num_features -= 1
        for line in lines:
            line = line.split(',')
            line[-1] = line[-1][:-1]
            line = [int(x) for x in line] 
            line.insert(0, 1)
            if netflix_or_heart():
                line.pop(-2)
            num_lines += 1
            y = line[-1]
            if y == 1:
                count_y_1 += 1
            else:
                count_y_0 += 1
     # account for pushing a 1 to the beginning
    lst_thetas = initialize_list(num_features + 1)

    for i in range(N_STEPS):
        gradient = initialize_list(num_features + 1)

        for line in lines:
            weighted_sum = 0
            line = line.split(',')
            line[-1] = line[-1][:-1]
            line = [int(x) for x in line] 
            line.insert(0, 1)
            y = line[-1]
            print(line)
            for j in range(len(lst_thetas)):
                weighted_sum += (lst_thetas[j] * line[j])
                print(line[j])

            for k in range(len(gradient)):
                gradient[k] += line[k]*(y - sigmoid(weighted_sum))
        for x in range(len(lst_thetas)):
            lst_thetas[x] += STEP_SIZE * gradient[x]
    
    # TESTING 
    with open(TEST_FILE) as file:
        df = pd.read_csv(TRAIN_FILE)
        cols = len(df.axes[1]) 
        num_features = cols - 1
        lines = file.readlines()[1:]
        num_lines = len(lines)
        correct = 0
        for line in lines:
            weighted_sum = 0
            line = line.split(',')
            line[-1] = line[-1][:1]
            line = [int(x) for x in line] 
            line.insert(0, 1)
            y = line[-1]
            for j in range(len(lst_thetas)):
                weighted_sum += (lst_thetas[j] * line[j])
             # argmax of the weighted sum, we check to see if it is more likely that Y = 1 or Y = 0 based on probability that Y <= 0.5 (Y=0 in this case)
            probability = sigmoid(weighted_sum) 
            prediction = 1
            if probability <= 0.5:
                prediction = 0
            if prediction == y:
                correct += 1
    print("Number of Correct: ", correct)
    print("Number of Tests:", num_lines)
    print("Classification Accuracy: ", correct / num_lines)

training()