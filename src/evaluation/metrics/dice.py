def dice_similarity(list1, list2, print_intersection=False):
    '''
    Calculates Dice Coefficient between two lists of words
    '''
    set1 = set(list1)
    set2 = set(list2)
    intersection = len(list(set1.intersection(list2)))
    if print_intersection:
        print(list(set1.intersection(list2)))
#     union = (len(list1) + len(list2)) - intersection
    return 2*float(intersection / (len(set1)+len(set2)))


def calculate_dice_sim(R1,R2):
    '''
    Calculates Dice Coefficient for each sentence pair and returns nested list with similarities in each subset.
    :param R1: raw text 1
    :param R2: raw text 2
    :return : nested list with dice coefficent
    '''
    subset_overlap = []
    for n in range(len(R1)):
        sim_per_pair = []
        for i in range(len(R1[n])):
            s1 = R1[n][i]
            s2 = R2[n][i]
            sim = dice_similarity(s1, s2)
            sim_per_pair.append(sim)
        subset_overlap.append(sim_per_pair)
    return subset_overlap


if __name__ == '__main__':
    l1 = ['this', 'is', 'great']
    l2 = ['this', 'not']

    dice_similarity(l1, l2, True)
