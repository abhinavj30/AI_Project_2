import numpy as np
import scipy.special


def file_read(file_type, file_name):
    data = []
    file = open(file_name, 'r')
    header_read = file.readline().split()
    if file_type == 0:
        num_inputs = int(header_read[1])

        for line in file:
            line = [float(word) for word in line.split()]
            data = data + [line]

        data_x = np.asarray(data).T[:num_inputs].T
        data_y = np.asarray(data).T[num_inputs:].T
        return data_x, data_y
    elif file_type == 1:
        num_input_nodes_read = int(header_read[0])
        num_hidden_nodes_read = int(header_read[1])
        num_output_nodes_read = int(header_read[2])

        w1_read = []
        for ii in range(num_hidden_nodes_read):
            line = [float(word) for word in file.readline().split()]
            w1_read = w1_read + [line]

        w1_read = np.asarray(w1_read).T

        w2_read = []
        for ii in range(num_output_nodes_read):
            line = [float(word) for word in file.readline().split()]
            w2_read = w2_read + [line]

        w2_read = np.asarray(w2_read).T
        return num_input_nodes_read, num_hidden_nodes_read, num_output_nodes_read, w1_read, w2_read


def dsig(in_val):
    return scipy.special.expit(in_val) * (1 - scipy.special.expit(in_val))


print("Enter weights file")
net_file = input()

print("Enter training data file")
train_data_file = input()

print("Enter output file")
output_file = input()
open(output_file, 'wb')

print("Enter number of epochs")
num_epochs = int(input())

print("Enter learning rate")
alpha = float(input())

num_input_nodes, num_hidden_nodes, num_output_nodes, w1, w2 = file_read(1, net_file)
xdat, ydat = file_read(0, train_data_file)

for epoch in range(num_epochs):
    for ii in range(len(xdat)):
        sample_input = xdat[ii: ii + 1, :]
        sample_output = ydat[ii: ii + 1, :]
        a1 = np.append(-np.ones((len(sample_input), 1)), sample_input, axis=1)
        ins2 = np.matmul(a1, w1)
        a2 = scipy.special.expit(ins2)
        a2 = np.append(-np.ones((len(a2), 1)), a2, axis=1)
        ins3 = np.matmul(a2, w2)
        a3 = scipy.special.expit(ins3)
        delta3 = dsig(ins3) * (sample_output - a3)
        delta2 = dsig(ins2) * np.matmul(delta3, w2[1:, ].T)
        w2 += alpha * np.matmul(a2.T, delta3)
        w1 += alpha * np.matmul(a1.T, delta2)

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

print("done...")
