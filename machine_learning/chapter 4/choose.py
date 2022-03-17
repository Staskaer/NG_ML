# ���㵱ǰ����ı�ǩ��ѡ����ѵ�����

def choose_best_attr(train, names, func):
    # ѡ����ѵķ�������
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
    # ���������б�ǩ�Ƿ���ȫ���
    y = D.iloc[:, -1]
    if y.sum() == len(D) or y.sum() == 0:
        return 1
    return 0


def compute_major_label(D):
    # �������������ı�ǩ
    # ��Ϊֻ�����ֱ�ǩ��0��1��������ͼ����Ƿ������Ŀ��һ������ж�
    y = D.iloc[:, -1]
    if y.sum() > len(y)/2:
        return r'good'
    return r'bad'


def compute_equal_attr(D, del_names):
    # �ж����������Ƿ���ȫһ��
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
