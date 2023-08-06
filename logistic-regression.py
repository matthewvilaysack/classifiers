import csv
import math

train_file_name = 'heart-train.csv'

test_file_name = 'heart-test.csv'

train_file = open(train_file_name)

lines = train_file.readlines()
#get rid of label line
lines.pop(0)
for i in range(len(lines)):
    # break into sublists
    lines[i] = lines[i].split(",")
    # remove \n from last element of sublists
    lines[i][-1] = lines[i][-1][:-1]
    #X_0 bias weight
    lines[i].insert(0, "1")
    
    #!!!! FOR NETFLIX AND HEART: REMOVE DEMOGRAPHIC COLUMN FROM EACH LINE
    if (train_file_name == 'netflix-train.csv' or train_file_name == 'heart-train.csv'):
        lines[i].pop(-2)


#count num var
num_var = len(lines[0]) - 1 
print("Input has this many varaibles: " + str(num_var))

NUM_RUNS = 1000
STEP = .00001

# NUM_RUNS = 100
# STEP = 0.00625

def get_weighted_avg(row):
    final_sum = 0
    for i in range(num_var):
        final_sum = final_sum + parameters[i] * float(row[i])
    return final_sum

def sigmoid(arg):
    return 1/(1 + math.exp(-1 * arg))

#for 3b
def get_log_likelihood(row):
    final_sum = 0
    for i in range(num_var):
        final_sum = final_sum + (float(row[-1])*math.log(sigmoid(get_weighted_avg(row))) + (1 - float(row[-1])) * math.log(1 - sigmoid(get_weighted_avg(row))))
    return final_sum

#initialize parameters to 0
parameters = []
for i in range(num_var):
    parameters.append(0)

#for 3b part ii
pre_train_log = 1.0
for row in lines:
    pre_train_log = pre_train_log * get_log_likelihood(row)
print("Pre Train Log Likelihood: " + str(pre_train_log))

for _ in range(NUM_RUNS):
    gradient = []
    #initalize to 0
    for i in range(num_var):
        gradient.append(0)

    for row in lines:
        weighted_avg = get_weighted_avg(row)
        for i in range(num_var):
            gradient[i] = gradient[i] + float(row[i])*(float(row[-1]) - sigmoid(weighted_avg))

    for i in range(num_var):
        parameters[i] = parameters[i] + (gradient[i] * STEP)

#for 3b part i
for i in range(num_var):
    print("for X" + str(i) + " the parameter is: " + str(parameters[i]))

#for 3c part iii
post_train_log = 1.0
for row in lines:
    post_train_log = post_train_log * get_log_likelihood(row)
print("Post Train Log Likelihood: " + str(post_train_log))

test_file = open(test_file_name)

test_lines = test_file.readlines()
#pop off label row
test_lines.pop(0)
#get rid of \n
for i in range(len(test_lines)):
    # break into sublists
    test_lines[i] = test_lines[i].split(",")
    #bias
    test_lines[i].insert(0, "1")
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
    p_Y1 = sigmoid(get_weighted_avg(row))
    final_result = 0

    if (p_Y1 > 0.5):
        final_result = 1
    else:
        final_result = 0

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
