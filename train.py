
# organize imports
from __future__ import print_function

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.svm import SVC
import numpy as np
import h5py
import os
import json
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.filterwarnings('ignore')

# load the user configs
with open('conf/conf.json') as f:    
	config = json.load(f)

# config variables
modle_name         = config["model"]
test_size 		= config["test_size"]
# seed 			= config["seed"]
features_path 	= config["features_path"]
labels_path 	= config["labels_path"]
results 		= config["results"]
classifier_path = config["classifier_path"]
train_path 		= config["train_path"]
num_classes 	= config["num_classes"]
classifier_path = config["classifier_path"]

# import features and labels
h5f_data  = h5py.File(features_path, 'r')
h5f_label = h5py.File(labels_path, 'r')

features_string = h5f_data['dataset_1']
labels_string   = h5f_label['dataset_1']

print(labels_string)


features = np.array(features_string)
labels   = np.array(labels_string)

h5f_data.close()
h5f_label.close()

# verify the shape of features and labels
print ("[INFO] features shape: {}".format(features.shape))
print ("[INFO] labels shape: {}".format(labels.shape))

print ("[INFO] training started...")
# split the training and testing data
# https://www.cnblogs.com/Yanjy-OnlyOne/p/11288098.html
(trainData, testData, trainLabels, testLabels) = train_test_split(np.array(features),
                                                                  np.array(labels),
                                                                  test_size=test_size) # 0.10
                                                                #   random_state=seed

print ("[INFO] splitted train and test data...")
print ("[INFO] train data  : {}".format(trainData.shape))
print ("[INFO] test data   : {}".format(testData.shape))
print ("[INFO] train labels: {}".format(trainLabels.shape))
print ("[INFO] test labels : {}".format(testLabels.shape))

# use logistic regression as the model
# https://blog.csdn.net/weixin_41712499/article/details/82526483
print ("[INFO] creating model %s-Logistic..."%(modle_name))
# model = svm.SVC(probability=False) 
# model = LogisticRegression(penalty='l1',solver='liblinear') #随机数种子 seed
model = LogisticRegression(penalty='l2',solver='liblinear')
model.fit(trainData, trainLabels)

# train_score=model.score(trainData, trainLabels) #回归问题：以R2参数为标准 分类问题：以准确率为标准
# text_score=model.score(testData,testLabels)
# print("[INFO] train_score=%f"%(train_score))
# print("[INFO] text_score=%f"%(text_score))

# sys.exit()


# dump classifier to file
# https://www.cnblogs.com/whiteprism/p/6201451.html
print ("[INFO] saving model%s-Logistic..."%(modle_name))
pickle.dump(model, open(classifier_path, 'wb'))

#---------------------------------------------------------------

# use rank-1 and rank-5 predictions
print ("[INFO] evaluating model %s-Logistic..."%(modle_name))
f = open(results, "w")
rank_1 = 0
rank_5 = 0

# loop over test data
for (label, features) in zip(testLabels, testData):
	# predict the probability of each class label and
	# take the top-5 class labels
	# https://blog.csdn.net/qq_38486203/article/details/80967696
	predictions = model.predict_proba(np.atleast_2d(features))[0]  
	#返回获得的8个种类的所有结果的概率
	predictions = np.argsort(predictions)[::-1][:5] 
	#输出x中最大的5项对应的index,并将其转化为数组

	# rank-1 prediction increment
	if label == predictions[0]:
		rank_1 += 1

	# rank-5 prediction increment
	if label in predictions:
		rank_5 += 1

# convert accuracies to percentages
rank_1 = (rank_1 / float(len(testLabels))) * 100
rank_5 = (rank_5 / float(len(testLabels))) * 100
print(test_size)
print(rank_1,rank_5)

# write the accuracies to file
f.write("Rank-1: {:.2f}%\n".format(rank_1))
f.write("Rank-5: {:.2f}%\n\n".format(rank_5))

# evaluate the model of test data
preds = model.predict(testData)

# write thprint (preds)e classification report to file
f.write("{}\n".format(classification_report(testLabels, preds)))
f.close()

# display the confusion matrix
print ("[INFO] confusion matrix")

# get the list of training lables
labels = sorted(list(os.listdir(train_path)))


# plot the confusion matrix  , labels=["butterfly", "cat","dog","elephant","horse","monkey","sheep","spider"]
cm = confusion_matrix(testLabels, preds)
sns.heatmap(cm,
            annot=True,
            cmap="Set2")
plt.show()