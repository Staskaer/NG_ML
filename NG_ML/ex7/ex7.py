# ��ϰ7

from get_data import get_data_k_means, get_data_pca_img, get_img, get_data_pca
from find_closest_centroids import find_closest_centroids
from init_centroids import init_centroids
from compute_centroids import compute_centroids
from reduce_date import recover_data, projetc_data
import numpy as np
import cv2
import matplotlib.pyplot as plt


def run_k_means(max_iter=20):
    # k-means�㷨ʵ�־���
    x = get_data_k_means()
    centroids = init_centroids(x, 3)

    m, n = x.shape
    idx = idx_ = np.zeros(m)

    for i in range(max_iter):
        idx = find_closest_centroids(x, centroids=centroids)
        centroids = compute_centroids(x, idx, 3)

        # ��ͼ
        cluster1 = x[np.where(idx == 0)[0], :]
        cluster2 = x[np.where(idx == 1)[0], :]
        cluster3 = x[np.where(idx == 2)[0], :]
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.scatter(cluster1[:, 0], cluster1[:, 1],
                   s=30, color='r', label='Cluster 1')
        ax.scatter(cluster2[:, 0], cluster2[:, 1],
                   s=30, color='g', label='Cluster 2')
        ax.scatter(cluster3[:, 0], cluster3[:, 1],
                   s=30, color='b', label='Cluster 3')
        ax.legend()
        plt.show()

        # ����ʱֹͣ
        if (idx == idx_).all():
            break
        idx_ = idx


def run_k_means_pic(max_iter=10):
    # ��k-means�㷨ʵ��ͼƬѹ��
    # �þ����ҵ���ߴ����Ե�������ɫ��
    # ����ԭʼ��24λͼ��ӳ�䵽��ά����ɫ�ռ�
    a = get_img()['A']
    a = a/255  # ��һ��
    x = np.reshape(a, (a.shape[0]*a.shape[1], a.shape[2]))

    # ִ��k-means�㷨
    # ѹ������λ���ɴ˴�ȷ��
    centroids = init_centroids(x, 16)
    m, n = x.shape
    idx = np.zeros(m)

    for i in range(max_iter):
        idx = find_closest_centroids(x, centroids=centroids)
        centroids = compute_centroids(x, idx, 3)

    x_recovered = centroids[idx.astype(int), :]
    x_recovered = np.reshape(x_recovered, (a.shape[0], a.shape[1], a.shape[2]))
    plt.imshow(x_recovered)
    plt.show()


def PCA():
    # ���ɷַ���
    data = get_data_pca()
    x = data['X']
    x = (x-x.mean())/x.std()  # ��һ��

    # ����Э����
    x = np.matrix(x)
    cov = (x.T*x)/x.shape[0]

    # svd�ֽ�
    U, S, V = np.linalg.svd(cov)

    # �ȶ�ԭʼ���ݽ�ά����ά
    z = projetc_data(x, U, 1)
    x_rec = recover_data(z, U, 1)
    # ��ͼ
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(list(x_rec[:, 0]), list(x_rec[:, 1]))
    plt.show()


def PCA_img():
    # ��pca����ͼ�����ݵĽ�ά
    x = get_data_pca_img()
    # ѡһ��ͼ����pca��ά
    face = np.reshape(x[3, :], (32, 32))
    plt.imshow(face)
    plt.show()
    # ��opencv��ʾЧ������
    # face_ = cv2.resize(face, (400, 400))
    # cv2.imshow("raw", face_)

    x = (x-x.mean())/x.std()  # ��һ��
    # ����Э����
    x = np.matrix(x)
    cov = (x.T*x)/x.shape[0]
    # svd�ֽ�
    U, S, V = np.linalg.svd(cov)

    z = projetc_data(x, U, 100)
    x_rec = recover_data(z, U, 100)

    face_Rec = np.reshape(x_rec[3, :], (32, 32))
    plt.imshow(face_Rec)
    plt.show()
    # face_r = cv2.resize(face_Rec, (400, 400))
    # cv2.imshow("rec", face_r)
    # cv2.waitKey(-1)


if __name__ == "__main__":
    PCA_img()
