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

#####################################
#           Configuration           #
#####################################
DISTRIBUTION			= 0
PERCEPTRON 				= 1
LOGISTICS_REGRESSION 	= 2
SVM						= 3

OPTIONS         		= 0
ALGORITHM				= PERCEPTRON
RANDOM_SPLIT			= False

RANDOM_STATE			= 133

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

	X_train_std, X_test_std, y_train, y_test = Preparation()

	if ALGORITHM == DISTRIBUTION:
		distribution( X_train_std, X_test_std, y_train, y_test )
	elif ALGORITHM == PERCEPTRON:
		tryPerceptron( X_train_std, X_test_std, y_train, y_test )
	elif ALGORITHM == LOGISTICS_REGRESSION:
		tryLogisticRegression( X_train_std, X_test_std, y_train, y_test )
	elif ALGORITHM == SVM:
		trySVM( X_train_std, X_test_std, y_train, y_test )


def getOptions(args, argc):
	global OPTIONS
	global ALGORITHM
	global RANDOM_STATE
	for index, arg in enumerate(args):
		if arg == "-o" or arg == "-O" and index < (argc-1) and '-' not in args[index+1]:
			OPTIONS = args[index+1].split(',')
		elif arg == '-a' or arg == '-A' and index < (argc-1) and '-' not in args[index+1]:
			ALGORITHM = int(args[index+1])
		elif arg == '-r' or arg == '-R' and index < (argc-1) and '-' not in args[index+1]:
			RANDOM_STATE = int(args[index+1])

def distribution( X_train_std, X_test_std, y_train, y_test ):

	X_combined_std = np.vstack((X_train_std, X_test_std))

	y_combined = np.hstack((y_train, y_test))

	plot_decision_regions(X=X_combined_std, y=y_combined, classifier=None,
						  test_idx=None,
						  title=u'あやめ特徴量の分布')


def Preparation():
	iris = datasets.load_iris()

#	print(iris)

	X = iris.data[:, [2,3]]	# シンプルにするため３列目と４列目の特徴量２つだけを取得する
	y = iris.target			# クラスラベル(結果として期待する値を取得)
	# これらはどちらもベクトルデータ

	print('Class labels:{a}'.format(a=np.unique(y)))	# np.unique()で重複を取り除き期待する結果の種類を表示する

	# トレーニングデータとテストデータに分割する。なぜならテストデータが無いので。
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RANDOM_STATE)	# 全体の30%をランダムにテストデータにする。

#	seprate_line = len(X)-20

#	X_train = X[:seprate_line]
#	X_test  = X[seprate_line:]

#	y_train = y[:seprate_line]
#	y_test  = y[seprate_line:]

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

	return X_train_std, X_test_std, y_train, y_test

def tryPerceptron( X_train_std, X_test_std, y_train, y_test ):

	# エポック数40、学習率0.1でパーセプトロンのインスタンスを生成
	ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0, shuffle=True)

	# トレーニングデータをモデルに適合させる
	ppn.fit(X_train_std, y_train)

	# テストデータで予測を実施
	y_pred = ppn.predict(X_test_std)

	# 誤分類のサンプルの個数を表示
	print('Misclassified samples: {a}'.format(a=(y_test != y_pred).sum()),flush=True)

	# 正解率を表示
	print('Accuracy: %.2f' % accuracy_score(y_test, y_pred),flush=True)

	X_combined_std = np.vstack((X_train_std, X_test_std))

	y_combined = np.hstack((y_train, y_test))

	plot_decision_regions(X=X_combined_std, y=y_combined, classifier=ppn,
						  test_idx=range(len(X_train_std),len(X_train_std)+len(X_test_std)),
						  title=u'パーセプトロンを使ったあやめの分類学習と予測')


def tryLogisticRegression( X_train_std, X_test_std, y_train, y_test ):

	lr = LogisticRegression(C=1000.0, random_state=0)

	lr.fit(X_train_std, y_train)

	y_pred = lr.predict(X_test_std)

	# 誤分類のサンプルの個数を表示
	print('Misclassified samples: {a}'.format(a=(y_test != y_pred).sum()),flush=True)

	# 正解率を表示
	print('Accuracy: %.2f' % accuracy_score(y_test, y_pred),flush=True)

	X_combined_std = np.vstack((X_train_std, X_test_std))

	y_combined = np.hstack((y_train, y_test))

	plot_decision_regions(X=X_combined_std, y=y_combined, classifier=lr,
						  test_idx=range(len(X_train_std),len(X_train_std)+len(X_test_std)),
						  title=u'ロジスティック回帰を使ったあやめの分類学習と予測')


def trySVM( X_train_std, X_test_std, y_train, y_test ):

	svm = SVC(kernel='linear', C=1.0, random_state=0)

	svm.fit(X_train_std, y_train)

	y_pred = svm.predict(X_test_std)

	# 誤分類のサンプルの個数を表示
	print('Misclassified samples: {a}'.format(a=(y_test != y_pred).sum()),flush=True)

	# 正解率を表示
	print('Accuracy: %.2f' % accuracy_score(y_test, y_pred),flush=True)

	X_combined_std = np.vstack((X_train_std, X_test_std))

	y_combined = np.hstack((y_train, y_test))

	plot_decision_regions(X=X_combined_std, y=y_combined, classifier=svm,
						  test_idx=range(len(X_train_std),len(X_train_std)+len(X_test_std)),
						  title=u'サポートベクタマシンを使ったあやめの分類学習と予測')


def plot_decision_regions(X, y, classifier=None, test_idx=None, resolution=0.02, title=None):

	markers = ('s', 'x', 'o', '^', 'v')
	colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
	cmap = ListedColormap(colors[:len(np.unique(y))])

	fp = FontProperties(fname=r'C:\WINDOWS\Fonts\meiryo.ttc', size=14)

	iris_name = ['Hiougi Ayame (0)','Blue Flag (1)', 'Kakitsubata (2)']

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

	print('test_idx:{a}'.format(a=test_idx),flush=True)

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
