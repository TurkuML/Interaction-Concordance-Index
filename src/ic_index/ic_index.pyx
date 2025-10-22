include "swapped.pxi"

import numpy as np
cimport numpy as np

import math

np.import_array()

# round key to significant_digits digits
# this affects how close labels need to be to one another to be considered ties
# set significant_digits=-1 to skip rounding
def round_key(double k, int significant_digits):
    if significant_digits == -1:
        return k

    cdef int tolerance
    cdef int magnitude

    if abs(k) == 0.0:
        magnitude = 0
    else:
        magnitude = math.floor(math.log10(abs(k)))

    tolerance = significant_digits - magnitude

    return round(k, tolerance)


def count_ties(double[:] my_list, significant_digits):
    #We get the number of ties in linear time by using a hash table and looking at duplicates
    #k duplicates means k(k-1)/2 additional tied pairs, e.g. in [1,2,3,3,4,3,4,5] you have
    #4 tied pairs, (3,3) thrice and (4,4) once
    cdef dict freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1

    cdef int ties = 0
    for value in freq.values():
        ties += value*(value-1)/2

    return ties


def ic_index(const np.int64_t[:] ID_dim1, const np.int64_t[:] ID_dim2, const double[:] labels, predictions, double rtol = 1e-14, double atol = 1e-14, significant_digits=14):

    # Swap the lists of IDs so that dim1 has fewer elements than dim2.
    if len(set(ID_dim1)) > len(set(ID_dim2)):
        temp = ID_dim1
        ID_dim1 = ID_dim2
        ID_dim2 = temp

    cdef np.ndarray dim1_array = np.array(ID_dim1)
    cdef np.ndarray dim2_array = np.array(ID_dim2)
    cdef np.ndarray labels_array = np.array(labels)
    cdef np.ndarray predictions_array
    cdef int n_models
    if predictions.ndim == 1:
        predictions_array = np.asarray(predictions).reshape(-1, 1)
        n_models = 1#.shape[1]
    else:
        predictions_array = np.array(predictions)
        n_models = predictions_array.shape[1]


    cdef set dim1s = set(ID_dim1)
    cdef set dim2s = set(ID_dim2)
    cdef int element1, element2, ties, n
    cdef int n_dim1 = len(dim1s)
    cdef double pairs = 0.0
    cdef double[:] discordant = np.zeros(shape = n_models)
    cdef double s
    cdef set dim2s_element1, dim2s_element2, dim2s_element1_element2

    for i_element1 in range(n_dim1-1):
        element1 = list(dim1s)[i_element1]
        ind_element1 = np.where(np.equal(ID_dim1, element1))[0]
        
        # Elements in dim2 whose interactions with element1 are known.
        dim2s_ind_element1 = dim2_array[ind_element1]
        dim2s_element1 = set(dim2s_ind_element1)

        for i_element2 in range(i_element1+1, n_dim1):
            element2 = list(dim1s)[i_element2]
            ind_element2 = np.where(np.equal(ID_dim1, element2))[0]
            # Elements in dim2 whose interactions with element2 are known.
            dim2s_ind_element2 = dim2_array[ind_element2]
            dim2s_element2 = set(dim2s_ind_element2)

            dim2s_element1_element2 = dim2s_element1.intersection(dim2s_element2)
            
            ind_element1_dim2Common = ind_element1[np.where(list(element in dim2s_element1_element2 for element in dim2s_ind_element1))]
            ind_element2_dim2Common = ind_element2[np.where(list(element in dim2s_element1_element2 for element in dim2s_ind_element2))]

            n = len(dim2s_element1_element2)

            Y_element1 = labels_array[ind_element1_dim2Common]
            Y_element2 = labels_array[ind_element2_dim2Common]
            y_differences = Y_element1 - Y_element2

            ties = count_ties(y_differences, significant_digits)

            for m in range(n_models):    
                P_element1 = predictions_array[ind_element1_dim2Common,m]
                P_element2 = predictions_array[ind_element2_dim2Common,m]
                p_differences = P_element1 - P_element2
                
                s = count_swapped(y_differences, p_differences, rtol, atol)
                discordant[m] += s

            pairs += n*(n-1)/2 - ties

    if predictions.ndim == 1:
        return 1.0 - discordant[0] / pairs
    else:
        return 1.0 - np.array(discordant) / pairs
