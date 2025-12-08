import math

import numpy as np
def method_difference(scores1,scores2):
    difference = scores1-scores2
    n = len(difference)
    mae = np.nansum(abs(difference))/ n
    mean_error = np.nansum(difference) / n
    std = math.sqrt(np.nansum((difference-mean_error)**2)/n)
    rmse = math.sqrt(np.nansum(difference**2)/ n)
    statistc_data = {
        "MAE" : mae,
        "STD" : std,
        "RMSE" : rmse
    }
    return statistc_data