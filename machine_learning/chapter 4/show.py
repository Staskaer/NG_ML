# ���ƾ�������ʹ���˻���ѧϰʵս�еĴ���

import matplotlib.pylab as plt
import matplotlib

# �ܹ���ʾ����
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['font.serif'] = ['SimHei']

# �ֲ�ڵ㣬Ҳ���Ǿ��߽ڵ�
decisionNode = dict(boxstyle="sawtooth", fc="0.8")

# Ҷ�ӽڵ�
leafNode = dict(boxstyle="round4", fc="0.8")

# ��ͷ��ʽ
arrow_args = dict(arrowstyle="<-")


def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    # """
    # ����һ���ڵ�
    # :param nodeTxt: �����ýڵ���ı���Ϣ
    # :param centerPt: �ı�������
    # :param parentPt: ������꣬����Ҳ��ָ���ڵ������
    # :param nodeType: �ڵ�����,��ΪҶ�ӽڵ�;��߽ڵ�
    # :return:
    # """

    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                            xytext=centerPt, textcoords='axes fraction',
                            va="center", ha="center", bbox=nodeType, arrowprops=arrow_args)


def getNumLeafs(myTree):
    # """
    # ��ȡҶ�ڵ����Ŀ
    # :param myTree:
    # :return:
    # """
    # ͳ��Ҷ�ӽڵ������
    numLeafs = 0

    # �õ���ǰ��һ��key��Ҳ���Ǹ��ڵ�
    firstStr = list(myTree.keys())[0]

    # �õ���һ��key��Ӧ������
    secondDict = myTree[firstStr]

    # �ݹ����Ҷ�ӽڵ�
    for key in secondDict.keys():
        # ���key��Ӧ����һ���ֵ䣬�͵ݹ����
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        # ���ǵĻ���˵����ʱ��һ��Ҷ�ӽڵ�
        else:
            numLeafs += 1
    return numLeafs


def getTreeDepth(myTree):
    # """
    # �õ�������Ȳ���
    # :param myTree:
    # :return:
    # """
    # ��������������
    maxDepth = 0

    # �õ����ڵ�
    firstStr = list(myTree.keys())[0]

    # �õ�key��Ӧ������
    secondDic = myTree[firstStr]

    # ���������ӽڵ�
    for key in secondDic.keys():
        # ����ýڵ����ֵ䣬�͵ݹ����
        if type(secondDic[key]).__name__ == 'dict':
            # �ӽڵ����ȼ�1
            thisDepth = 1 + getTreeDepth(secondDic[key])

        # ˵����ʱ��Ҷ�ӽڵ�
        else:
            thisDepth = 1

        # �滻������
        if thisDepth > maxDepth:
            maxDepth = thisDepth

    return maxDepth


def plotMidText(cntrPt, parentPt, txtString):
    # """
    # ��������ڵ���ӽڵ���м�λ�ã������Ϣ
    # :param cntrPt: �ӽڵ�����
    # :param parentPt: ���ڵ�����
    # :param txtString: �����ı���Ϣ
    # :return:
    # """
    # ����x����м�λ��
    xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
    # ����y����м�λ��
    yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
    # ���л���
    createPlot.ax1.text(xMid, yMid, txtString)


def plotTree(myTree, parentPt, nodeTxt):
    # """
    # ���Ƴ��������нڵ㣬�ݹ����
    # :param myTree: ��
    # :param parentPt: ���ڵ������
    # :param nodeTxt: �ڵ���ı���Ϣ
    # :return:
    # """
    # ����Ҷ�ӽڵ���
    numLeafs = getNumLeafs(myTree=myTree)

    # �����������
    depth = getTreeDepth(myTree=myTree)

    # �õ����ڵ����Ϣ����
    firstStr = list(myTree.keys())[0]

    # �������ǰ���ڵ��������ӽڵ���м�����,Ҳ���ǵ�ǰx���ƫ�������ϼ�������ĸ��ڵ������λ����Ϊx�ᣨ����˵��һ�Σ���ʼ��xƫ����Ϊ��-1/2W,��������ĸ��ڵ�����λ��Ϊ��(1+W)/2W����ӵõ���1/2������ǰy��ƫ������Ϊy��
    cntrPt = (plotTree.xOff + (1.0 + float(numLeafs)) /
              2.0/plotTree.totalW, plotTree.yOff)

    # ���Ƹýڵ��븸�ڵ����ϵ
    plotMidText(cntrPt, parentPt, nodeTxt)

    # ���Ƹýڵ�
    plotNode(firstStr, cntrPt, parentPt, decisionNode)

    # �õ���ǰ���ڵ��Ӧ������
    secondDict = myTree[firstStr]

    # ������µ�y��ƫ�����������ƶ�1/D��Ҳ������һ��Ļ���y��
    plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD

    # ѭ���������е�key
    for key in secondDict.keys():
        # �����ǰ��key���ֵ�Ļ�����������������ݹ����
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            # �����µ�x��ƫ������Ҳ�����¸�Ҷ�ӻ��Ƶ�x�����������ƶ���1/W
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            # ��ע�Ϳ��Թ۲�Ҷ�ӽڵ������仯
            # print((plotTree.xOff, plotTree.yOff), secondDict[key])
            # ����Ҷ�ӽڵ�
            plotNode(secondDict[key], (plotTree.xOff,
                                       plotTree.yOff), cntrPt, leafNode)
            # ����Ҷ�ӽڵ�͸��ڵ���м���������
            plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))

    # ���صݹ�֮ǰ����Ҫ��y���ƫ�������ӣ������ƶ�1/D��Ҳ���Ƿ���ȥ������һ���y��
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
    # """
    # ��Ҫ���Ƶľ�����
    # :param inTree: �������ֵ�
    # :return:
    # """
    # ����һ��ͼ��
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    # ��������������ܿ��
    plotTree.totalW = float(getNumLeafs(inTree))
    # ������������������
    plotTree.totalD = float(getTreeDepth(inTree))
    # ��ʼ��x��ƫ������Ҳ����-1/2W��ÿ�������ƶ�1/W��Ҳ���ǵ�һ��Ҷ�ӽڵ���Ƶ�x����Ϊ��1/2W���ڶ�����3/2W����������5/2W�����һ����(W-1)/2W
    plotTree.xOff = -0.5/plotTree.totalW
    # ��ʼ��y��ƫ������ÿ�����»��������ƶ�1/D
    plotTree.yOff = 1.0
    # ���ú������л��ƽڵ�ͼ��
    plotTree(inTree, (0.5, 1.0), '')
    # ����
    plt.show()


# testTree = {'no surfacing': {0: 'no', 1: {
#     'flippers': {0: 'no', 1: 'yes'}}, 3: 'maybe'}}
# createPlot(testTree)
