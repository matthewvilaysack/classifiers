import csv

train_file_name = 'netflix-train.csv'

test_file_name = 'netflix-test.csv'

train_file = open(train_file_name)

lines = train_file.readlines()
#get rid of label line
lines.pop(0)
for i in range(len(lines)):
    # break into sublists
    lines[i] = lines[i].split(",")
    # remove \n from last element of sublists
    lines[i][-1] = lines[i][-1][:-1]
    #!!!!!!!!FOR NETFLIX AND HEART: REMOVE DEMOGRAPHIC COLUMN FROM EACH LINE
    if (train_file_name == 'netflix-train.csv' or train_file_name == 'heart-train.csv'):
        lines[i].pop(-2)
#count num var
num_var = len(lines[0]) - 1
print("num_var", num_var)

print("Input has this many varaibles: " + str(num_var))

#first find P(Y = 1), start variables at 1 and 2 for laplace smoothing (MAKE 0 FOR NUMERIC ANSWER)
p_Y1 = 0
total_rows = 0

for row in lines:
    #if the last column, Y, is 1
    if row[-1] == "1":
        p_Y1 = p_Y1 + 1
    total_rows = total_rows + 1

#calculate p_Y1, p_Y0
p_Y1 = p_Y1 / total_rows
p_Y0 = 1 - p_Y1
print("P(Y = 1) = " + str(p_Y1))

#array of probability X_i = 1 | Y = 1
p_X1_Y1 = []
#array of probability X_i = 0 | Y = 1
p_X0_Y1 = []

#for each X_i, calculate P(X_i = 1 | Y = 1)
for i in range(num_var):
    #numerator set to 1 for laplace smoothing
    n_X1_Y1 = 1
    #denominator set to 2 for laplace smoothing
    n_Y1 = 2
    for row in lines:
        # if Y = 1
        if (row[-1] == "1"):
            n_Y1 = n_Y1 + 1
            # if Y = 1 and X_i = 1
            if (row[i] == "1"):
                n_X1_Y1 = n_X1_Y1 + 1
    p_X1_Y1.append(n_X1_Y1 / n_Y1)
    p_X0_Y1.append(1 - (n_X1_Y1 / n_Y1))

#array of probability X_i = 1 | Y = 0
p_X1_Y0 = []
#array of probability X_i = 0 | Y = 0
p_X0_Y0 = []

#for each X_i, calculate P(X_i = 1 | Y = 0)
for i in range(num_var):
    #numerator set to 1 for laplace smoothing
    n_X1_Y0 = 1
    #denominator set to 2 for laplace smoothing
    n_Y0 = 2
    for row in lines:
        # if Y = 0
        if (row[-1] == "0"):
            n_Y0 = n_Y0 + 1
            # if Y = 1 and X_i = 1
            if (row[i] == "1"):
                n_X1_Y0 = n_X1_Y0 + 1
    p_X1_Y0.append(n_X1_Y0 / n_Y0)
    p_X0_Y0.append(1 - (n_X1_Y0 / n_Y0))

test_file = open(test_file_name)

test_lines = test_file.readlines()
#pop off label row
test_lines.pop(0)
#get rid of \n
for i in range(len(test_lines)):
    # break into sublists
    test_lines[i] = test_lines[i].split(",")
    # remove \n from last element of sublists
    test_lines[i][-1] = test_lines[i][-1][:-1]


#used for accuracy
n_success = 0
n_Y1_correct = 0
n_Y1 = 0
n_Y0_correct = 0 
n_Y0 = 0
n_trials = 0


for row in test_lines:
    likelihood_Y0 = 1.0
    likelihood_Y1 = 1.0
    for i in range(num_var):
        if row[i] == "1":
            likelihood_Y0 = likelihood_Y0 * float(p_X1_Y0[i])
            likelihood_Y1 = likelihood_Y1 * float(p_X1_Y1[i])
        else:
            likelihood_Y0 = likelihood_Y0 * float(p_X0_Y0[i])
            likelihood_Y1 = likelihood_Y1 * float(p_X0_Y1[i])
    final_result = 0
    # print(likelihood_Y0 * p_Y0)
    # print(likelihood_Y1 * p_Y1)
    # print('\n')
    if (likelihood_Y0 * p_Y0 > likelihood_Y1 * p_Y1):
        final_result = 0
    else:
        final_result = 1

    if (row[-1] == "0"):
        n_Y0 = n_Y0 + 1
        if final_result == 0:
            n_Y0_correct = n_Y0_correct + 1
    else:
        n_Y1 = n_Y1 + 1
        if final_result == 1:
            n_Y1_correct = n_Y1_correct + 1
    
    if str(final_result) == row[-1]:
        n_success = n_success + 1
    n_trials = n_trials + 1

print("Class 0: tested " + str(n_Y0) + ", correctly classified " + str(n_Y0_correct))
print("Class 1: tested " + str(n_Y1) + ", correctly classified " + str(n_Y1_correct))
print("Overall: tested " + str(n_trials) + ", correctly classified " + str(n_success))
print("Accuracy: " + str(n_success / n_trials))

#for 2b part 3
#for i in range(num_var):
#  print("for X" + str(i + 1) + " the ratio is: " + str((p_X1_Y1[i]*p_Y1) / (p_X1_Y0[i]*p_Y0)))




