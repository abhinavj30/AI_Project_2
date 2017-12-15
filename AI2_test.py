import numpy as np
import scipy.special
from sklearn.metrics import confusion_matrix


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

print("Enter test data file")
test_data_file = input()

print("Enter output file")
output_file = input()
open(output_file, 'wb')

num_input_nodes, num_hidden_nodes, num_output_nodes, w1, w2 = file_read(1, net_file)
xdat, ydat = file_read(0, test_data_file)

a1 = np.append(-np.ones((len(xdat), 1)), xdat, axis=1)
ins2 = np.matmul(a1, w1)
a2 = scipy.special.expit(ins2)
a2 = np.append(-np.ones((len(a2), 1)), a2, axis=1)
ins3 = np.matmul(a2, w2)
a3 = scipy.special.expit(ins3)

y_calc = np.round(a3, 0)
sample_results = np.zeros((num_output_nodes, 8))
micro_results = np.zeros(4)
macro_results = np.zeros(4)
sum_results = np.zeros(4)

for ii in range(num_output_nodes):
    confusion_sample = np.reshape(confusion_matrix(ydat[:, ii], y_calc[:, ii]), 4)
    sample_results[ii, 0] = confusion_sample[3]
    sample_results[ii, 1] = confusion_sample[1]
    sample_results[ii, 2] = confusion_sample[2]
    sample_results[ii, 3] = confusion_sample[0]

    sum_results[0] += confusion_sample[0]
    sum_results[1] += confusion_sample[1]
    sum_results[2] += confusion_sample[2]
    sum_results[3] += confusion_sample[3]

    sample_results[ii, 4] = (confusion_sample[3] + confusion_sample[0]) / (np.sum(confusion_sample))
    sample_results[ii, 5] = confusion_sample[3] / (confusion_sample[3] + confusion_sample[1])
    sample_results[ii, 6] = confusion_sample[3] / (confusion_sample[3] + confusion_sample[2])
    sample_results[ii, 7] = (2 * sample_results[ii, 5] * sample_results[ii, 6]) / (sample_results[ii, 5] + sample_results[ii, 6])

micro_results[0] = (sum_results[3] + sum_results[0]) / (np.sum(sum_results))
micro_results[1] = sum_results[3] / (sum_results[3] + sum_results[1])
micro_results[2] = sum_results[3] / (sum_results[3] + sum_results[2])
micro_results[3] = (2 * micro_results[1] * micro_results[2]) / (micro_results[1] + micro_results[2])

macro_results[0] = np.average(sample_results[:, 4], axis=0)
macro_results[1] = np.average(sample_results[:, 5], axis=0)
macro_results[2] = np.average(sample_results[:, 6], axis=0)
macro_results[3] = (2 * macro_results[1] * macro_results[2]) / (macro_results[1] + macro_results[2])

file_handler = open(output_file, 'w')
for ii in range(sample_results.shape[0]):
    result_string = '%d %d %d %d %0.3f %0.3f %0.3f %0.3f\n' % tuple(sample_results[ii, :])
    file_handler.write(result_string)
micro_string = '%0.3f %0.3f %0.3f %0.3f\n' % tuple(micro_results)
file_handler.write(micro_string)
macro_string = '%0.3f %0.3f %0.3f %0.3f\n' % tuple(macro_results)
file_handler.write(macro_string)

print("done...")
