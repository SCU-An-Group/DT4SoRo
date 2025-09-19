import numpy as np
import subprocess
from os import path
import os

curr_dir = os.getcwd()
training_set_dir = curr_dir + '//FEModelFiles_training_set'
test_set_dir = curr_dir + '//FEModelFiles_test_set'
if not os.path.exists(training_set_dir):
    os.mkdir(training_set_dir)
if not os.path.exists(test_set_dir):
    os.mkdir(test_set_dir)

num_training_data = 17
num_test_data = 2
num_sum = num_training_data + num_test_data

grid_P = np.linspace(0.0, 0.09, 240)
selected_P = np.random.choice(grid_P, size=num_sum, replace=False)
np.random.shuffle(selected_P)
P_values = np.round(selected_P[:num_training_data], 6)
test_P_values = np.round(selected_P[num_training_data:], 6)

with open("Parameters_P.py", "w") as f:
    f.write("P_values = " + str(P_values.tolist()) + '\n')
    f.write("test_P_values = " + str(test_P_values.tolist()) + '\n')

params = {}
with open(os.path.join(curr_dir, "Parameters_P.py")) as f:
    exec(f.read(), {}, params)
P_values = params["P_values"]
test_P_values = params["test_P_values"]

for i in range(len(P_values)):
    with open(os.path.join(curr_dir, "Parameters_for_Fpn_bending_Gravity.py"), "w") as f:
        f.write("P = %f\n" % P_values[i])
    
    try:
        subprocess.call([r'runAba.bat'])
    except Exception as e:
        print("job %d failed: %s" % (i+1, str(e)))


for i in range(len(test_P_values)):
    with open(os.path.join(curr_dir, "Parameters_for_Fpn_bending_Gravity.py"), "w") as f:
        f.write("P = %f\n" % test_P_values[i])
    
    try:
        subprocess.call([r'runAba.bat'])
    except Exception as e:
        print("job %d failed: %s" % (i+1, str(e)))