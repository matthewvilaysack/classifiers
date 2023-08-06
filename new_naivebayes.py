
# import util
import numpy as np
import pandas as pd
import math

TRAIN_FILE = 'heart-train.csv'
TEST_FILE = 'heart-test.csv'


# Training Algorithm on simple-test.csv
def training():

    training_file = open(TRAIN_FILE)
    lines = training_file.readlines()[1:] # skip first line
    count_y_0 = 0
    count_y_1 = 0
    num_features = 0
    x_1_y_1 = [0] * num_features # List containing a count for eevery feature x_1,x_2,...x_n. The case where x_i = 1 and y = 1
    x_0_y_0 = [0] * num_features # List containing a count for eevery feature x_1,x_2,...x_n. The case where x_i = 0 and y = 0
    x_1_y_0= [0] * num_features # List containing a count for eevery feature x_1,x_2,...x_n. The case where x_i = 1 and y = 0
    x_0_y_1 = [0] * num_features # List containing a count for eevery feature x_1,x_2,...x_n. The case where x_i = 0 and y = 1
    num_lines = 0
    lines = []
    file = training_file.readlines()[1:]
    for i in range(len(lines)):
        # sublist for each count for x_i
        lines[i] = lines[i].split(",")
        # remove the new line character
        lines[i][-1] = lines[i][-1][:-1]
        # special case for netflix and heart files, remove demographic
        if (TRAIN_FILE == 'netflix-small-train.csv' or TRAIN_FILE == 'heart-train.csv'):
            lines[i].pop(-2)
        y = lines[i][-1]
        if y == 1:
            count_y_1 += 1
        else:
            count_y_0 += 1
        num_features = len(lines[0]) - 1
        


        for i in range(num_features):
            x_i = lines[i]
    
            if x_i == 1 and y == 1:
                x_1_y_1[i] += 1
            elif x_i == 0 and y == 0:
                x_0_y_0[i] += 1
            elif x_i == 1 and y == 0:
                x_1_y_0[i] += 1
            else:
                x_0_y_1[i] += 1
    print("num features", num_features)
    # print(x_0_y_0)
    # print(x_1_y_1)
    # print(x_1_y_0)
    # print(x_0_y_1)

    # prob_x_1_y_1 = []
    # prob_x_0_y_1 = []

    # # For each x_i, calculate P(X_i = 1 | Y = 1):
    # for i in range(num_features):
    #     #numerator set to 1 for laplace smoothing
    #     n_x_1_y_1 = 1
    #     #denominator set to 2 for laplace smoothing
    #     n_Y1 = 2
    #     for row in lines:
    #         # if Y = 1
    #         if (row[-1] == "1"):
    #             n_YY += 1
    #             # if Y = 1 and X_i = 1
    #             if (row[i] == "1"):
    #                 n_x_1_y_1 += 1
    #     prob_x_1_y_1.append(n_x_1_y_1 / n_Y1)
    #     prob_x_0_y_1.append(1 - (n_x_1_y_1 / n_Y1))

    # # prob_x_1_y_1 = [(num + 1) / (count_y_1 + 2) for num in x_1_y_1];
    # prob_x_0_y_0 = [(num + 1) / (count_y_0 + 2) for num in x_0_y_0];
    # prob_x_1_y_0 = [(num + 1) / (count_y_0 + 2) for num in x_1_y_0];
    # prob_x_0_y_1 = [(num + 1) / (count_y_1 + 2) for num in x_0_y_1];
    # prob_y_1 = (count_y_1 + 1) / (num_lines + 2)
    # prob_y_0 = (count_y_0 + 1) / (num_lines + 2)    
    # # print(prob_x_0_y_0)
    # # print(prob_x_1_y_0)
    # # print(prob_x_0_y_1)
    # # print(prob_x_1_y_1)
    # # print(prob_y_1)
    # #TESTING

    # predictions = []
    # correct = 0
    # n_success = 0
    # num_lines = 0

    # with open(TEST_FILE) as file:
    #     # Using the naive bayes assumption we want the summation of argmax(log(P(Y=y)) + sum of logs of P(Y=y | x_1, x_2, ... , x_n)
    #     file = file.readlines()[1:]
    #     for line in file:
    #         likelihood_Y1 = 1.0
    #         likelihood_Y0 = 1.0
    #         P_Y_1_X = math.log(prob_y_1) # P(Y=1|x_1,..,x_n)
    #         P_Y_0_X = math.log(prob_y_0) # P(Y=0|x_1,..,x_n)
    #         num_lines += 1
    #         line = line.split(',')
    #         y = float(line[2][0])

    #         for i in range(0, num_features):
    #             x_i = float(line[i])

    #             if x_i == 1:
    #                 likelihood_Y0 *= float(prob_x_1_y_0[i])
    #                 likelihood_Y1 *= float(prob_x_1_y_1[i])
    #                 P_Y_1_X += math.log(prob_x_1_y_1[i])
    #                 P_Y_0_X += math.log(prob_x_1_y_0[i])
    #             else:
    #                 likelihood_Y0 *= float(prob_x_0_y_0[i])
    #                 likelihood_Y1 *= float(prob_x_0_y_1[i])
    #                 P_Y_1_X += math.log(prob_x_0_y_1[i])
    #                 P_Y_0_X += math.log(prob_x_0_y_0[i])
    #         # print("P_Y_0_X: ", P_Y_0_X)
    #         # print("P_Y_1_X: ", P_Y_1_X)
    #         # print('\n')
    #         # We want to get the argmax of Y, so we compare P_Y_0_X to P_Y_1_X
    #         final_result = 0
    #         # print(likelihood_Y0 * prob_y_0)
    #         # print(likelihood_Y1 * prob_y_1)
    #         # print('\n')
    #         if P_Y_1_X < P_Y_0_X:
    #             predictions.append(0)
    #             final_result = 0
    #             if y == 0:
    #                 correct += 1
    #         else:
    #             predictions.append(1)
    #             final_result = 1
    #             if y == 1:
    #                 correct += 1
    #         if final_result == y:
    #             n_success += 1
    #     # print("num lines: ", num_lines)
            
    #     print("Predictions: ", predictions)
        
    #     # print("Class 0: tested " + str(n_Y0) + ", correctly classified " + str(n_Y0_correct))
    #     # print("Class 1: tested " + str(n_Y1) + ", correctly classified " + str(n_Y1_correct))
    #     print("Overall: tested " + str(num_lines) + ", correctly classified " + str(n_success))
    #     print("Number correct: ", correct)
    #     print("Number correct / Total Entries: ", (correct/(num_lines))) 

    #     # PART B: NETFLIX
    #     # print("P(Y=1): " ,prob_y_1)


    
    

    # # print(num_features)
                
            




training()

