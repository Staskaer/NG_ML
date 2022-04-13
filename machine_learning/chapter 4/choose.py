# 计算当前子类的标签、选出最佳的属性

def choose_best_attr(train, names, func):
    # 选出最佳的分类属性
    latest = 0
    best = 0
    best_attr = ''
    for a in names:
        latest = func(train, a)
        if latest < best:
            best = latest
            best_attr = a
    return best_attr


def judge_equal(D):
    # 计算子类中标签是否完全相等
    y = D.iloc[:, -1]
    if y.sum() == len(D) or y.sum() == 0:
        return 1
    return 0


def compute_major_label(D):
    # 计算子类中最多的标签
    # 因为只有两种标签，0或1，所以求和计算是否大于数目的一半就能判断
    y = D.iloc[:, -1]
    if y.sum() > len(y)/2:
        return r'good'
    return r'bad'


def compute_equal_attr(D, del_names):
    # 判断样本属性是否完全一致
    x = D.iloc[:, :-1]

    x = x.drop(del_names, axis=1)
    flag = 1

    index = list(x.index)
    for i in range(1, len(index)):
        try:
            if not x.loc[index[i-1]].equals(x.loc[index[i]]):
                flag = 0
                break
        except:
            pass
    return flag
