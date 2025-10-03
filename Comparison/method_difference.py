def method_difference(scores1,scores2):
    difference = abs(scores1-scores2)
    total = sum(sum(difference))/(len(difference)*len(difference[0]))
    return total