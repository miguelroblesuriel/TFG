import numpy as np

def get_hits_standard(scores, comparison_table):
    hits = 0
    rank = 0
    i = 0
    for row in scores:
        j = 0
        indices = np.argsort(row)
        indices_desc = indices[::-1]
        for index in indices_desc:
            if index in comparison_table[i] and j == 0:
                hits += 1
                j = j + 1
                break
            elif index in comparison_table[i]:
                break
            if j == 10:
                break
            j = j + 1
        rank += j
        i = i + 1
    rank = rank/i
    return hits, rank




