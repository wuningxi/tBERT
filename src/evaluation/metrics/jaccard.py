def jaccard_similarity(list1, list2, print_intersection=False):
    '''
    Calculates Jaccard Index between two lists of words
    '''
    intersection = len(list(set(list1).intersection(list2)))
    if print_intersection:
        print(list(set(list1).intersection(list2)))
    union = (len(list1) + len(list2)) - intersection
    return float(intersection / union)


def calculate_jaccard_index(R1,R2):
    '''
    Calculates Jaccard Index for each sentence pair and returns nested list with similarities in each subset.
    :param R1: raw text 1
    :param R2: raw text 2
    :return : nested list with jaccard index
    '''
    subset_overlap = []

    for n in range(len(R1)):
        sim_per_pair = []
        for i in range(len(R1[n])):
            s1 = R1[n][i]
            s2 = R2[n][i]
            sim = jaccard_similarity(s1,s2)
            sim_per_pair.append(sim)
        subset_overlap.append(sim_per_pair)
    return subset_overlap

if __name__=='__main__':
    l1 = ['this','is','great']
    l2 = ['this','not']
    print(jaccard_similarity(l1, l2, True))