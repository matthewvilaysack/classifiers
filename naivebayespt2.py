
# import util
import numpy as np
import pandas as pd
import math

TRAIN_FILE = 'heart-train.csv'
TEST_FILE = 'heart-test.csv'

def netflix_or_heart():
    if (TRAIN_FILE == 'netflix-train.csv' or TRAIN_FILE == 'heart-train.csv'):
        return True
    return False
# Training Algorithm on simple-test.csv
def training():
    df = pd.read_csv(TRAIN_FILE)
    cols = len(df.axes[1]) 
    num_features = cols - 1
    count_y_0 = 0
    count_y_1 = 0
    x_1_y_1 = [0] * num_features # List containing a count for eevery feature x_1,x_2,...x_n. The case where x_i = 1 and y = 1
    x_0_y_0 = [0] * num_features # List containing a count for eevery feature x_1,x_2,...x_n. The case where x_i = 0 and y = 0
    x_1_y_0= [0] * num_features # List containing a count for eevery feature x_1,x_2,...x_n. The case where x_i = 1 and y = 0
    x_0_y_1 = [0] * num_features # List containing a count for eevery feature x_1,x_2,...x_n. The case where x_i = 0 and y = 1
    num_lines = 0
    with open(TRAIN_FILE) as file:
        file = file.readlines()[1:]
        if netflix_or_heart():
            num_features -= 1
        for line in file:
            line = line.split(',')
            line[-1] = line[-1][:-1]
            line = [int(x) for x in line] 
            if netflix_or_heart():
                line.pop(-2)
            num_lines += 1
            y = line[-1]
            if y == 1:
                count_y_1 += 1
            else:
                count_y_0 += 1

            for i in range(num_features):
                x_i = int(line[i])
                if x_i == 1 and y == 1:
                    x_1_y_1[i] += 1
                elif x_i == 0 and y == 0:
                    x_0_y_0[i] += 1
                elif x_i == 1 and y == 0:
                    x_1_y_0[i] += 1
                else:
                    x_0_y_1[i] += 1

    prob_x_1_y_1 = [(num + 1) / (count_y_1 + 2) for num in x_1_y_1];
    prob_x_0_y_0 = [(num + 1) / (count_y_0 + 2) for num in x_0_y_0];
    prob_x_1_y_0 = [(num + 1) / (count_y_0 + 2) for num in x_1_y_0];
    prob_x_0_y_1 = [(num + 1) / (count_y_1 + 2) for num in x_0_y_1];
    prob_y_1 = (count_y_1 + 1) / (num_lines + 2)
    prob_y_0 = (count_y_0 + 1) / (num_lines + 2)    
    # For part 1b
    for i in range(len(prob_x_1_y_1)):
        print("x_i where i = ", i)
        print("P(X_i = 1 | Y = 1): ", prob_x_1_y_1[i])

    predictions = []
    correct = 0
    with open(TEST_FILE) as file:
        # Using the naive bayes assumption we want the summation of argmax(log(P(Y=y)) + sum of logs of P(Y=y | x_1, x_2, ... , x_n)
        file = file.readlines()[1:]
        # print(file)
        num_lines = 0
        for line in file:
            P_Y_1_X = math.log(prob_y_1) # P(Y=1|x_1,..,x_n)
            P_Y_0_X = math.log(prob_y_0) # P(Y=0|x_1,..,x_n)
            num_lines += 1
            line = line.split(',')
            line[-1] = line[-1][:1]
            if netflix_or_heart():
                line.pop(-2)
        
            line = [int(x) for x in line] 

            y = line[-1]

            for i in range(num_features):
                x_i = line[i]

                if x_i == 1:
                    P_Y_1_X += math.log(prob_x_1_y_1[i])
                    P_Y_0_X += math.log(prob_x_1_y_0[i])
                else:
                    P_Y_1_X += math.log(prob_x_0_y_1[i])
                    P_Y_0_X += math.log(prob_x_0_y_0[i])
            # We want to get the argmax of Y, so we compare P_Y_0_X to P_Y_1_X
            if P_Y_1_X <= P_Y_0_X:
                predictions.append(0)
                if y == 0:
                    correct += 1
            else:
                predictions.append(1)
                if y == 1:
                    correct += 1

        print("Number correct: ", correct)
        print("Number of tests", num_lines)
        print("Number correct / Total Entries: ", (correct/(num_lines))) 



    
    

                
            




training()

