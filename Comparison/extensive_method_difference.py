import math
from Comparison.method_difference import method_difference
import numpy as np
def extensive_method_difference(scores1,scores2):
    scores1_flat = scores1.flatten()
    scores2_flat = scores2.flatten()
    sort_order = np.argsort(scores1_flat)
    scores1_flat = scores1_flat[sort_order]
    scores2_flat = scores2_flat[sort_order]
    scores1_first_matrix = []
    scores1_second_matrix = []
    scores1_third_matrix = []
    scores1_fourth_matrix = []
    scores1_fifth_matrix = []
    scores2_first_matrix = []
    scores2_second_matrix = []
    scores2_third_matrix = []
    scores2_fourth_matrix = []
    scores2_fifth_matrix = []
    for i in range(len(scores1_flat)):
        if scores1_flat[i] <= 0.2:
            scores1_first_matrix.append(scores1_flat[i])
            scores2_first_matrix.append(scores2_flat[i])
        elif scores1_flat[i] <= 0.4:
            scores1_second_matrix.append(scores1_flat[i])
            scores2_second_matrix.append(scores2_flat[i])
        elif scores1_flat[i] <= 0.6:
            scores1_third_matrix.append(scores1_flat[i])
            scores2_third_matrix.append(scores2_flat[i])
        elif scores1_flat[i] <= 0.8:
            scores1_fourth_matrix.append(scores1_flat[i])
            scores2_fourth_matrix.append(scores2_flat[i])
        else:
            scores1_fifth_matrix.append(scores1_flat[i])
            scores2_fifth_matrix.append(scores2_flat[i])
    print("0-0.2:")
    print(method_difference(np.array(scores1_first_matrix),np.array(scores2_first_matrix)))
    print("0.2-0.4:")
    print(method_difference(np.array(scores1_second_matrix),np.array(scores2_second_matrix)))
    print("0.4-0.6:")
    print(method_difference(np.array(scores1_third_matrix),np.array(scores2_third_matrix)))
    print("0.6-0.8:")
    print(method_difference(np.array(scores1_fourth_matrix),np.array(scores2_fourth_matrix)))
    print("0.8-1.0:")
    print(method_difference(np.array(scores1_fifth_matrix),np.array(scores2_fifth_matrix)))
