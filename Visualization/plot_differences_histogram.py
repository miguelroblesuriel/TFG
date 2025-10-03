
def plot_differences_histogram(reference, scores = None):
    for score in scores:
        difference = abs(reference - score)
        values = difference.flatten()
        print(values)

