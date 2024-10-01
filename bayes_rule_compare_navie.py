import numpy as np
import pandas as pd
import scipy


#导入数据
#使用scipy.io加载.mat文件
mat_data_test = scipy.io.loadmat(r"C:\Users\LinYi\Downloads\data_test.mat")
mat_data_train = scipy.io.loadmat(r"C:\Users\LinYi\Downloads\data_train.mat")
mat_label_train = scipy.io.loadmat(r"C:\Users\LinYi\Downloads\label_train.mat")
data_test = mat_data_test['data_test']
data_train = mat_data_train['data_train']
label_train = mat_label_train['label_train']
#将数据转换为Dataframe
data_test = pd.DataFrame(data_test)
data_train = pd.DataFrame(data_train)
label_train = pd.DataFrame(label_train)


# 定义计算多元正态分布的函数
def multivariate_gaussian(X, mean, cov):
    d = len(X)  # 特征的维度
    cov_inv = np.linalg.inv(cov)  # 计算协方差矩阵的逆
    det_cov = np.linalg.det(cov)  # 计算协方差矩阵的行列式
    
    # 计算多元正态分布概率密度函数值
    term1 = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det_cov))
    term2 = np.exp(-0.5 * (X - mean).T @ cov_inv @ (X - mean))
    
    return term1 * term2


#求先验概率，均值，协方差
def seprateByClass(label_data, data_train):
    #哪些类
    classes = label_data[label_data.columns[0]].unique()
    class_count = label_data.value_counts()
    class_prior = class_count/len(label_data) #先验概率
    
    #均值
    data_mean_0 = np.mean(data_train[(label_data==classes[0]).values], axis=0)
    data_mean_1 = np.mean(data_train[(label_data==classes[1]).values], axis=0)
    data_mean = np.array([data_mean_0,data_mean_1])
    #协方差
    data_cov_0 = np.cov(data_train[(label_data == classes[0]).values], rowvar= False)
    data_cov_1 = np.cov(data_train[(label_data == classes[1]).values], rowvar= False)
    data_cov = np.array([data_cov_0, data_cov_1])
    print(data_cov.shape)
    return classes, class_prior, data_mean, data_cov


# 定义 Bayes 决策规则
def bayes_decision_rule(X):
    #求训练数据集中的均值和协方差
    classes, class_prior, data_mean, data_cov =seprateByClass(label_train, data_train)
    
    # 计算 P(X | C1) 和 P(X | C2)
    # #def multivariate_gaussian(label_data, classes, data_train, mean, cov)
    likelihood_C1 = multivariate_gaussian(X, data_mean[0], data_cov[0])
    likelihood_C2 = multivariate_gaussian(X, data_mean[1], data_cov[1])
    
    # 计算后验概率 P(C1 | X) 和 P(C2 | X)
    posterior_C1 = likelihood_C1 * class_prior.iloc[0]
    posterior_C2 = likelihood_C2 * class_prior.iloc[1]
    
    # 比较后验概率，选择概率最大的类别
    if posterior_C1 > posterior_C2:
        return classes[0]
    else:
        return classes[1]


#print(data_test.iloc[0])
# 测试data_test
# 进行分类并输出结果
pred_bayes=[]
for index, row in data_test.iterrows():
    decision = bayes_decision_rule(row)
    print(f"For X = {row}, the decision is: {decision}")
    pred_bayes.append(decision)
print('the dicisions using bayes decision rule:', pred_bayes)
print('------------------------------------------------------------------------')

##采用朴素贝叶斯进行预测
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_data = data_train.values
test_data = data_test.values
train_label = label_train.values

# 创建并训练模型
model = GaussianNB()
model.fit(train_data, train_label.ravel())

#得到训练的均值和方差
means = model.theta_
variances = model.var_

#进行预测
y_pred = model.predict(test_data)
print('the dicisions using naive bayes rule:', y_pred.tolist())

#判断两次结果是否相同
same = (np.array(pred_bayes) == y_pred).all()
print('两个结果是否相同： ', same)