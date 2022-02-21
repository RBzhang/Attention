import numpy as np

def make_k_fold(k , num):
    r = np.random.RandomState(2)
    r.shuffle(num)
    idx = [int(i) for i in np.floor(np.linspace(0, len(num), k + 1))]
    # print(idx)
    train_fold = []
    test_fold = []
    for i in range(k):
        test = num[idx[i]:idx[i+1]]
        test_fold.append(test)
        train = np.setdiff1d(num, test)
        r.shuffle(train)
        train_fold.append(train)
    return train_fold, test_fold