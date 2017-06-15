# -*- coding: utf-8 -*-
import os
import sys
import subprocess
from datetime import datetime

import random

import numpy as np

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

from matplotlib.font_manager import FontProperties

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

#####################################
#           Configuration           #
#####################################
SAMPLE					= 0
PERCEPTRON 				= 1
LOGISTICS_REGRESSION 	= 2
SVM						= 3
DISTRIBUTION1			= 10
DISTRIBUTION2			= 11

OPTIONS         		= 0
ALGORITHM				= PERCEPTRON
RANDOM_SPLIT			= False

MAX_NUMBER_OF_SAMPLES	= 1500

# Meta
# X : 特徴量
# y : 結果変数(クラスラベル)
# w : 重み
# fit : 近似処理

#####################################
#            subroutines            #
#####################################
def main(args, argc):

	getOptions(args, argc)

	X_train_std, X_test_std, y_train, y_test, y_images = Preparation()

	if ALGORITHM == SAMPLE:
		sample()
	elif ALGORITHM == DISTRIBUTION1:
		distribution(1)
	elif ALGORITHM == DISTRIBUTION2:
		distribution(2)
	elif ALGORITHM == PERCEPTRON:
		tryPerceptron( X_train_std, X_test_std, y_train, y_test, y_images )
	elif ALGORITHM == LOGISTICS_REGRESSION:
		tryLogisticRegression( X_train_std, X_test_std, y_train, y_test, y_images )
	elif ALGORITHM == SVM:
		trySVM( X_train_std, X_test_std, y_train, y_test, y_images )

def getOptions(args, argc):
	global OPTIONS
	global ALGORITHM
	for index, arg in enumerate(args):
		if arg == "-o" or arg == "-O" and index < (argc-1) and '-' not in args[index+1]:
			OPTIONS = args[index+1].split(',')
		elif arg == '-a' or arg == '-A' and index < (argc-1) and '-' not in args[index+1]:
			ALGORITHM = int(args[index+1])

def sample():
	digits = datasets.load_digits(10)

	images_and_predictions = list(zip(digits.images[:10],digits.target[:10]))
	for index, (image, prediction) in enumerate(images_and_predictions):
		plt.subplot(2, 5, index + 1)
		plt.axis('off')
		plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
		plt.title('Label: {a}'.format(a=prediction), size=12)
	plt.show()

def distribution(type):

	digits = datasets.load_digits(10)

	if type == 1:
		X_reduced = TSNE(n_components=2, random_state=0).fit_transform(digits.data)
	else:
		X_reduced = PCA(n_components=2).fit_transform(digits.data)
	plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=digits.target)
	plt.colorbar()
	plt.show()

def Preparation():
	global MAX_NUMBER_OF_SAMPLES

	digits = datasets.load_digits(10)
	MAX_NUMBER_OF_SAMPLES = len(digits.data)

	"""
	for index in range(0,5):
		for index2 in range(0,8):
			print('{a} {b} {c} {d} {e} {f} {g} {h}'.format(
					a = str(int(digits.data[index][index2+0])).rjust(4),
					b = str(int(digits.data[index][index2+1])).rjust(4),
					c = str(int(digits.data[index][index2+2])).rjust(4),
					d = str(int(digits.data[index][index2+3])).rjust(4),
					e = str(int(digits.data[index][index2+4])).rjust(4),
					f = str(int(digits.data[index][index2+5])).rjust(4),
					g = str(int(digits.data[index][index2+6])).rjust(4),
					h = str(int(digits.data[index][index2+7])).rjust(4)),
					flush=True)
		print('This is "{a}"'.format(a=digits.target[index]), flush=True)
	"""

	print('Class labels:{a}'.format(a=np.unique(digits.target)))	# np.unique()で重複を取り除き期待する結果の種類を表示する

#	seprate_line = int(MAX_NUMBER_OF_SAMPLES * -3 / 10)
	seprate_line = MAX_NUMBER_OF_SAMPLES-60

	X_train = digits.data[:seprate_line]
	X_test  = digits.data[seprate_line:]

	y_train = digits.target[:seprate_line]
	y_test  = digits.target[seprate_line:]

	y_images = digits.images[seprate_line:]

	print('Number of train:{a}'.format(a=len(X_train)))
	print('Number of test:{a}'.format(a=len(X_test)))

#	print('X_train:\r\n{a}'.format(a=X_train))	# トレーニング用特徴量データ
#	print('X_test:\r\n{a}'.format(a=X_test))	# テスト用特徴量データ
#	print('y_train:\r\n{a}'.format(a=y_train))	# トレーニング用クラスラベル
#	print('y_test:\r\n{a}'.format(a=y_test))	# テスト用クラスラベル

	# 重み付けがデータの絶対的な大きさに依存しないように取得した特徴量データを標準化してスケーリング。
	sc = StandardScaler()

	# トレーニングデータの平均と標準偏差を計算
	sc.fit(X_train)	# fitというのは"《数学》フィッティング◆実験などで得られた曲線を多項式などの数式で近似すること"

	# 平均と標準偏差を用いて標準化
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test)

	return X_train_std, X_test_std, y_train, y_test, y_images


def tryPerceptron( X_train_std, X_test_std, y_train, y_test, y_images ):

	# エポック数40、学習率0.1でパーセプトロンのインスタンスを生成
	ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0, shuffle=True)

	# トレーニングデータをモデルに適合させる
	ppn.fit(X_train_std, y_train)

	# テストデータで予測を実施
	y_pred = ppn.predict(X_test_std)

	print('y_pred:{a}'.format(a=y_pred),flush=True)

	# 誤分類のサンプルの個数を表示
	print('Misclassified samples: {a}'.format(a=(y_test != y_pred).sum()),flush=True)

	# 正解率を表示
	print('Accuracy: %.2f' % accuracy_score(y_test, y_pred),flush=True)

	images_and_predictions = list(zip(y_images, y_pred, y_test))
	for index, (image, prediction, correct) in enumerate(images_and_predictions[-(12*5):]):
		plt.subplot(5, 12, index + 1)
		plt.axis('off')
		plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
		if prediction != correct:
			color = 'red'
		else:
			color = 'black'
		plt.title('Prediction: {a} \n Correct: {b}'.format(a=prediction,b=correct), size=9, color=color)
	plt.show()
	"""
	X_combined_std = np.vstack((X_train_std, X_test_std))

	y_combined = np.hstack((y_train, y_test))

	plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
						  test_idx=range(len(X_train_std),len(X_train_std)+len(X_test_std)),
						  title=u'パーセプトロンを使ったあやめの分類学習と予測')
	"""


def tryLogisticRegression( X_train_std, X_test_std, y_train, y_test, y_images ):

	lr = LogisticRegression(C=1000.0, random_state=0)

	lr.fit(X_train_std, y_train)

	y_pred = lr.predict(X_test_std)

	# 誤分類のサンプルの個数を表示
	print('Misclassified samples: {a}'.format(a=(y_test != y_pred).sum()),flush=True)

	# 正解率を表示
	print('Accuracy: %.2f' % accuracy_score(y_test, y_pred),flush=True)

	images_and_predictions = list(zip(y_images, y_pred, y_test))
	for index, (image, prediction, correct) in enumerate(images_and_predictions[-(12*5):]):
		plt.subplot(5, 12, index + 1)
		plt.axis('off')
		plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
		if prediction != correct:
			color = 'red'
		else:
			color = 'black'
		plt.title('Prediction: {a} \n Correct: {b}'.format(a=prediction,b=correct), size=9, color=color)
	plt.show()

	"""
	X_combined_std = np.vstack((X_train_std, X_test_std))

	y_combined = np.hstack((y_train, y_test))

	plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr,
						  test_idx=range(len(X_train_std),len(X_train_std)+len(X_test_std)),
						  title=u'ロジスティック回帰を使ったあやめの分類学習と予測')
	"""


def trySVM( X_train_std, X_test_std, y_train, y_test, y_images ):

	svm = SVC(kernel='linear', C=1.0, random_state=0)

	svm.fit(X_train_std, y_train)

	y_pred = svm.predict(X_test_std)

	# 誤分類のサンプルの個数を表示
	print('Misclassified samples: {a}'.format(a=(y_test != y_pred).sum()),flush=True)

	# 正解率を表示
	print('Accuracy: %.2f' % accuracy_score(y_test, y_pred),flush=True)

	images_and_predictions = list(zip(y_images, y_pred, y_test))

	for index, (image, prediction, correct) in enumerate(images_and_predictions[-(12*5):]):
		plt.subplot(5, 12, index + 1)
		plt.axis('off')
		plt.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
		if prediction != correct:
			color = 'red'
		else:
			color = 'black'
		plt.title('Prediction: {a} \n Correct: {b}'.format(a=prediction,b=correct), size=9, color=color)
	plt.show()
	"""
	X_combined_std = np.vstack((X_train_std, X_test_std))

	y_combined = np.hstack((y_train, y_test))

	plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm,
						  test_idx=range(len(X_train_std),len(X_train_std)+len(X_test_std)),
						  title=u'サポートベクタマシンを使ったあやめの分類学習と予測')
	"""

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02, title=None):

	markers = ('s', 'x', 'o', '^', 'v','s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	fp = FontProperties(fname=r'C:\WINDOWS\Fonts\meiryo.ttc', size=14)

	iris_name = ['1','2','3','4','5','6','7','8','9','10',]

	x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
						   np.arange(x2_min, x2_max, resolution))

	if classifier is not None:
		Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
		Z = Z.reshape(xx1.shape)

		plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)

		plt.xlim(xx1.min(), xx1.max())
		plt.ylim(xx2.min(), xx2.max())

	for idx, cl in enumerate(np.unique(y)):
		plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
					alpha=0.8, c=cmap(idx),
#					marker=markers[idx], label=cl)
					marker=markers[idx], label=iris_name[cl])
	if test_idx:
		X_test, y_test = X[test_idx, :], y[test_idx]
		plt.scatter(X_test[:, 0], X_test[:, 1], alpha=0.5, linewidth=0.5, marker='^', c='yellow', s=55, label='test set')

	if title is not None:
		plt.title(title,fontproperties=fp,ha='center')

#	plt.xlabel('petal length [standardized]')
#	plt.ylabel('petal width [standardized]')
	plt.xlabel(u'ながさ [標準化済み]', fontproperties=fp)
	plt.ylabel(u'はば [標準化済み]', fontproperties=fp)

	plt.legend(loc='upper left')

	plt.show()


##########################
###        main        ###
##########################
main( args=sys.argv, argc=len(sys.argv) )
