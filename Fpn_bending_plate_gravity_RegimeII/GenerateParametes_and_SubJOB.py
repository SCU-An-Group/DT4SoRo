import numpy as np
from scipy.stats import qmc
import subprocess
import os
import matplotlib.pyplot as plt

curr_dir = os.path.dirname(os.path.abspath(__file__))
training_set_dir = os.path.join(curr_dir, 'FEModelFiles_training_set')
test_set_dir = os.path.join(curr_dir, 'FEModelFiles_test_set')
if not os.path.exists(training_set_dir):
    os.mkdir(training_set_dir)
if not os.path.exists(test_set_dir):
    os.mkdir(test_set_dir)

num_training_data = 85
num_test_data = 2
num_sum = num_training_data + num_test_data

p_bounds = [0.00, 0.09]
dh_bounds = [10, 50]

sampler = qmc.LatinHypercube(d=2, optimization="random-cd")
samples = sampler.random(n=num_sum)

l_bounds = [p_bounds[0], dh_bounds[0]]
u_bounds = [p_bounds[1], dh_bounds[1]]
scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

P_samples = scaled_samples[:, 0]
detaH_samples = scaled_samples[:, 1]

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20
plt.figure(figsize=(8, 6))
plt.scatter(P_samples, detaH_samples, color='red', marker='o', label='Sample Points')
plt.xlabel(r'$P$', fontsize=24)
plt.ylabel(r'$\Delta H$', fontsize=24)
plt.title('Latin hypercube sampling', fontsize=24)
plt.xlim(p_bounds[0], p_bounds[1])
plt.ylim(dh_bounds[0], dh_bounds[1])
# plt.legend(fontsize=20, loc='upper right')
plt.grid(True)
img_path = os.path.join(curr_dir, 'Latin hypercube sampling.png')
plt.savefig(img_path, dpi=300, bbox_inches='tight')
plt.show()

training_P    = np.round(P_samples[:num_training_data], 6)
training_detaH = np.round(detaH_samples[:num_training_data], 6)
test_P        = np.round(P_samples[num_training_data:], 6)
test_detaH    = np.round(detaH_samples[num_training_data:], 6)

param_file = os.path.join(curr_dir, "Parameters_P_detaH.py")
with open(param_file, "w") as f:
    f.write("P_values = " + str(training_P.tolist()) + '\n')
    f.write("detaH_values = " + str(training_detaH.tolist()) + '\n')
    f.write("test_P_values = " + str(test_P.tolist()) + '\n')
    f.write("test_detaH_values = " + str(test_detaH.tolist()) + '\n')

for i in range(len(training_P)):
    P = training_P[i]
    detaH = training_detaH[i]
    param_run_file = os.path.join(curr_dir, "Parameters_for_Fpn_bending_Board.py")
    with open(param_run_file, "w") as f:
        f.write("P = " + "%g" % P + '\n')
        f.write("detaH = " + "%g" % detaH + '\n')
    try:
        subprocess.call([r'runAba.bat'])
    except Exception as e:
        print("unexpected error!!!!!", e)

for i in range(len(test_P)):
    P = test_P[i]
    detaH = test_detaH[i]
    param_run_file = os.path.join(curr_dir, "Parameters_for_Fpn_bending_Board.py")
    with open(param_run_file, "w") as f:
        f.write("P = " + "%g" % P + '\n')
        f.write("detaH = " + "%g" % detaH + '\n')
    try:
        subprocess.call([r'runAba.bat'])
    except Exception as e:
        print("unexpected error!!!!!", e)
