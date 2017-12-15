from random import *
import numpy as np

output_file = "data.init"
num_input_nodes = 68
num_hidden_nodes = 5
num_output_nodes = 2
w1 = np.zeros((num_input_nodes + 1, num_hidden_nodes))
w2 = np.zeros((num_hidden_nodes + 1, num_output_nodes))

for ii in range (num_input_nodes + 1):
    for jj in range (num_hidden_nodes):
        w1[ii, jj] = random()*0.3

for ii in range (num_hidden_nodes + 1):
    for jj in range (num_output_nodes):
        w2[ii, jj] = random()*0.3


file_handler = open(output_file, 'w')
header = '%d %d %d\n' % (num_input_nodes, num_hidden_nodes, num_output_nodes)
file_handler.write(header)
for ii in range(num_hidden_nodes):
    buffer = ""
    for jj in range(w1.shape[0]):
        buffer += "%.3f " % (w1.T[ii, jj])
    buffer += "\n"
    file_handler.write(buffer)

for ii in range(num_output_nodes):
    buffer = ""
    for jj in range(w2.shape[0]):
        buffer += "%.3f " % (w2.T[ii, jj])
    buffer += "\n"
    file_handler.write(buffer)