import math
import numpy as np

def acc_ndcg(y_test, y_pred):
    acc = []
    for i in range(len(y_pred)):
        p = y_pred[i]
        t = y_test[i]
        try:
            index = int(np.where(p == t)[0][0]+1)
            dcg = 1/(math.log2(index + 1))
            acc.append(dcg)
        except:
            acc.append(0.0)
    acc = np.array(acc)
    return np.mean(acc)


        

def acc_ndcg_2(y_test, y_pred):
    try:
        index = int(np.where(y_pred == t)[0][0]+1)
        dcg = 1/(math.log2(index + 1))
    except:
        dcg = 0
    return dcg

t = [7, 2, 7, 7, 3, 4, 1]
p = np.array([[7, 2, 3, 4, 5], [7, 2, 3, 4, 5], [7, 2, 3, 4, 5], [7, 2, 3, 4, 5], [7, 2, 3, 4, 5], [7, 2, 3, 4, 5], [7, 2, 3, 4, 5]])

dcg = acc_ndcg(t, p)
print(dcg)
