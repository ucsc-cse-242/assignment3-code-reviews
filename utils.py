from numpy import ndarray, power, sqrt, sum as np_sum

def _euclidean_distance(x: ndarray, y: ndarray) -> ndarray:
    """
    find the euclidean distance between data and centroid

    inputs:
        @param data: numpy 2D array of data observations
        @param centroid: numpy array representing one point
    output:
        list of distances from observations to centroid

    Assumption: data observations and centroid must be same dimension
    """
    return sqrt(np_sum(power(x - y, 2), axis=1))
