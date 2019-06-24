import numpy as np


def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA-vecB,2)))

def randCent(dataSet, k):  #中心点初始化
    n = np.shape(dataSet)[1]
    centroids = np.mat(np.zeros([k, n]))  #创建 k 行 n列的全为0 的矩阵
    for j in range(n):
        minj = np.min(dataSet[:,j]) #获得第j 列的最小值
        rangej = float(np.max(dataSet[:,j]) - minj)     #得到最大值与最小值之间的范围
        #获得输出为 K 行 1 列的数据，并且使其在数据集范围内
        centroids[:,j] = np.mat(minj + rangej * np.random.rand(k, 1))
    return centroids

# np.random.seed(666) #定义一个随机种子
# rand_num = np.random.rand(3,1)  #输出为3行1 列,随机数在 0 到 1 之间
# print(rand_num)

def KMeans(dataSet,k,distMeans = distEclud,createCent=randCent):
    m = np.shape(dataSet)[0]
    clusterAssement = np.mat(np.zeros([m,2]))
    centroids = createCent(dataSet,k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            for j in range(k):
                distJ = distMeans(centroids[j,:],dataSet[i,:])
                if distJ<minDist:
                    minDist = distJ
                    minIndex = j
            if clusterAssement[i,0] !=minIndex:
                clusterChanged = True
            clusterAssement[i,:] = minIndex,minDist**2
        # print(centroids)

        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssement[:,0].A == cent)[0]]
            centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids,clusterAssement

print('---------- test KMeans ---------')
dataSet = np.mat([[ 0.90796996 ,5.05836784],[-2.88425582 , 0.01687006],
                    [-3.3447423 , -1.01730512],[-0.32810867 , 0.48063528]
                    ,[ 1.90508653 , 3.530091  ]
                    ,[-3.00984169 , 2.66771831]
                    ,[-3.38237045 ,-2.9473363 ]
                    ,[ 2.22463036 ,-1.37361589]
                    ,[ 2.54391447 , 3.21299611]
                    ,[-2.46154315 , 2.78737555]
                    ,[-3.38237045 ,-2.9473363 ]
                    ,[ 2.8692781  ,-2.54779119]
                    ,[ 2.6265299  , 3.10868015]
                    ,[-2.46154315 , 2.78737555]
                    ,[-3.38237045 ,-2.9473363 ]
                    ,[ 2.80293085 ,-2.7315146 ]])

center, cluster = KMeans(dataSet, 2)
print('----')
print(center)
print('----')
print(cluster)
























































