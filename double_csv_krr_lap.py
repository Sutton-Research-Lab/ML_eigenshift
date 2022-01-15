## ML_eigenshift
## The goal of the notbook below is to adopt the so-called 'delta-learning' techniques where we will use the outputs of 
## the lower level calculations to predict the output of the higher level calculations in the framework of density 
## functional theory (DFT). In this particular work we are utilizing the output of the Perdew-Burke-Ernzerhof (PBE)
## generalized gradient approximation(GGA) type exchange-correlation approximation. We are using the ML techniques to 
## learn the shift in the eigen values E_{ijk} from PBE to predict the corresponding eigen values for HSE06 (higher level 
## hybrid functional). Here, ijk are spin,band, and kpoint index. We begin the notebook by importing the required libraries.
## As a pre-requisite, following modules or libraries are required to run the notebook. In case you don't have some or all
## of the modules below, first try installing them [eg. you can use something like (pip install pandas)].

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
#
import tarfile
import re
import sys
import random 
import operator 
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

## We are storing the orbital projection of each eigenstates from PBE calculations and single point HSE06 calculations 
## in a csv file named after the compound. Since there are 110, compounds there are as many csv files stored in a 
## compressed tar.gz file (beast_csv_files.tar.gz). We first untar it if not done so previously.

def main(csv_file1,csv_file2):        
	df1 = pd.read_csv(csv_file1)    
	df2 = pd.read_csv(csv_file2)
	columns=['1s','2s','2p','3s','3p','3d','4s','4p','4d','5s','5p','5d','6s','6p','EF_PBE','eigshift(eV)']
	#columns=['EF_PBE']
	keep_cols = columns # 
	#keep_cols = ['EF_PBE']
	#y_feat = 'eigshift(eV)'

	train_data_inp=df1[keep_cols].to_numpy()[:, :-1]
	test_data_inp=df2[keep_cols].to_numpy()[:, :-1]
	train_data_tar=df1[keep_cols].to_numpy()[:, -1].reshape(-1, 1)
	test_data_tar=df2[keep_cols].to_numpy()[:, -1].reshape(-1, 1)
	#number_of_rows=x_data_ini.shape[0]
	#random_indices=x_data_ini[np.random.randint(x_data_ini.shape[0], size=int(number_of_rows/20)), :]

	#x_data =random_indices[:, :-1]
	#y_data =random_indices[:, -1]


	print(np.shape(train_data_inp)) #x_data)
	print(np.shape(test_data_inp)) #y_data)

	# for i, j in zip(x_data, y_data):
	# 	print(np.shape(i), i, j)

	number = random.randint(1, 1000)
	print("random seed", number)
	rseed = random.seed(number)

	
	#test_size = 0.20 
	nfold = 5
	nthread = 2
	ker = 'laplacian'
	#ker = 'rbf'

	alphas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0] #np.logspace(-15, 5, 21, base=2) #  # #
	gammas = np.linspace(0.001,0.1, 20) #np.logspace(-15, 5, 21, base=2) 

	all_data_test = []
	all_data_train = []
	all_data_pred_test = []
	all_data_pred_train = []
	all_data_mae = []
	for g in gammas:
		x_train, x_test, y_train, y_test = train_data_inp,test_data_inp,train_data_tar,test_data_tar
		clf, clf.best_params_, y_train, y_test, y_train_pred, y_test_pred = train_model(x_test, y_test, x_train, y_train, nfold, nthread, in_alpha=alphas, in_gamma=[g], kernel=ker, rseed=rseed)

		print("Number of training samples", len(y_train))
		print("Number of test samples", len(y_test))
		all_data_test.append( y_test ) 
		all_data_train.append( y_train )
		all_data_pred_test.append( y_test_pred )
		all_data_pred_train.append( y_train_pred )
		print("RMSE", root_mean_squared_error(y_test, y_test_pred))
		diff = abs(y_test - y_test_pred)
		print("MAE", np.mean(diff))
		all_data_mae.append(np.mean(diff))


	min_index, min_value = min(enumerate(all_data_mae), key=operator.itemgetter(1))

	tmp_figname ='_'.join(["_train_nsamples="+str(len(y_train)), "ker="+ker, "MAE="+str(min_value)])
	figname = tmp_figname+".pdf"
	xlabel = r"DFT $\Delta eigen$ (eV)"
	ylabel = r"ML $\Delta eigen}$ (eV)"
	error_unit = "meV"
	# xlabel = r"DFT $E_{g}$ (eV)"
	# ylabel = r"ML$E_{g}$ (eV)"
	# error_unit = "eV"
	# ylim=[-0.1, 0.1]
	# xmax = np.max(list(x2)+list(x2))*1.1
	# xmin = np.min(list(x2)+list(x2))*0.85
	# xlim=[xmin, xmax]


	scatter_comp(all_data_train[min_index], all_data_pred_train[min_index], all_data_test[min_index], all_data_pred_test[min_index], error_unit, xlabel, ylabel, figname)
	#sys.stdout.close()

def mae(y_true, y_pred):
    return np.mean(abs(y_true-y_pred))

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

def train_model(x_test, y_test, x_train, y_train, nfold, nthread, in_alpha=np.logspace(-15, 5, 21, base=2), in_gamma=np.logspace(-15, 3, 19, base=2), kernel=None, rseed=None):

	if rseed:
		random.seed(rseed)
		np.random.seed(rseed)

	neg_root_mean_squared_error = make_scorer(root_mean_squared_error, greater_is_better=False)

	clf = GridSearchCV(KernelRidge(kernel=kernel), cv=nfold, n_jobs=nthread, verbose=1, scoring=neg_root_mean_squared_error, param_grid={"alpha":in_alpha, "gamma": in_gamma})

	clf.fit(x_train, y_train)

	y_train_pred = clf.predict(x_train)
	y_test_pred = clf.predict(x_test)
	print('clf_best_params\n')
	print(clf.best_params_)
	#print(clf.best_params_['alpha'])
	#print(clf.best_params_['gamma'])
	return clf, clf.best_params_, y_train, y_test, y_train_pred, y_test_pred

def scatter_comp(x1, y1, x2, y2, error_unit, xlabel, ylabel, plot_name):

	import os, sys
	import matplotlib as mpl
	import matplotlib.pyplot as plt
	import pandas as pd
	import numpy as np

	from sklearn.metrics import r2_score

	axis_width = 2
	mpl.rcParams['mathtext.default'] = 'regular'
	mpl.rcParams['axes.linewidth'] = axis_width
	mpl.rcParams['axes.unicode_minus']=False

	fontsize = 24
	markersize = 150
	train_color = 'gray'
	test_color = 'blue'
	train_mark = 'o'
	test_mark = '^'
	alpha = 0.2
	lw45 = 1.5
	axis_width=1.5


	figsize = (9.2,7)
	label_font = 24
	tick_font = 20
	leg_font = 20
	tick_len = 6
	#test_marker = '^'
	test_marker = 'o'
	train_marker = 'o'
	train_color = 'gray'
	test_color = 'blue' #'red'
	pt_alpha = 0.3
	pt_lw = 0
	pt_s = 80
	htp=0.1
	diag_color = 'black'
	diag_lw = axis_width
	diag_ls = '-'
	text_font = 20
	axis_width = 2

	leg_fancy = False
	# pt_alpha = 0.3
	# text_font = 32
	# letter_font = 32
	# leg_font = 18
	leg_frame = True
	leg_alpha = 1
	leg_loc = 'upper right'
	htp = 0.2
	leg_color = 'white'
	leg_edge_color = 'black'
	leg_shadow = False
	leg_lw = axis_width
	s = 150
	pt_lw = 0
	decision_lw = axis_width
	decision_ls = '--'
	decision_color = 'black'
	# tick_len = 8
	# label_font = 32
	# tick_font = 32

	# figsize = (9.2, 7)
	# fontsize = 24
	markersize = 80
	# train_color = 'gray'
	# test_color = 'blue'
	# train_mark = '^'
	# test_mark = 'o' 
	alpha = 0.6
	lw45 = 1.5
	axis_width=1.5

	
	#train_err = rmse(y1, x1)
	#test_err = rmse(y2, x2)
	#print(("RMSE TRAIN ERROR", train_err))

	train_err = mae(y1, x1)
	test_err = mae(y2, x2)

	print(("MAE TRAIN ERROR", train_err))
	print(("MAE TEST ERROR", test_err))

	fig = plt.figure(figsize=figsize)
	# plt.scatter(y_train_pred, y_train, color=train_color, marker=train_mark, s=markersize, alpha=alpha, lw=0)
	# plt.scatter(y_test_pred, y_test, color=test_color, marker=test_mark, s=markersize, alpha=alpha, lw=0)
	plt.scatter(x1, y1, color=train_color, marker=train_mark, s=markersize, alpha=alpha, lw=0)
	plt.scatter(x2, y2, color=test_color, marker=test_mark, s=markersize, alpha=alpha, lw=0)
	plt.legend(['train', 'test'], fontsize=fontsize-4, frameon=False, loc='lower right')
	
	# xmax = np.max(list(y_train)+list(y_test))*1.1
	# xmin = np.min(list(y_train)+list(y_test))*0.85
	#plt.xlim([xmin, xmax])
	#plt.ylim([xmin, xmax])
	xmax = np.max(x1)
	xmin = np.min(x1)
	x45 = np.linspace(xmax*.98, xmin*1.02, num=100)
	y45 = x45
	plt.plot(x45, y45, color='black', lw=lw45, label='__nolegend__')
	# plt.ylabel('actual formation energy (eV/atom)', fontsize=fontsize)
	# plt.xlabel('pred formation energy (eV/atom)', fontsize=fontsize)
	plt.ylabel(ylabel, fontsize=fontsize)
	plt.xlabel(xlabel, fontsize=fontsize)

	plt.xticks(fontsize=fontsize)
	plt.yticks(fontsize=fontsize)
	axes = plt.gca()
	axes.set_title(axes.get_title()) #* 2)
	axes.set_xlabel(axes.get_xlabel(), size=fontsize) #* 0.5) # fontname="Times New Roman")
	axes.set_ylabel(axes.get_ylabel(), size=fontsize)
	#axes.text(.01,.90, plot_label, fontsize=fontsize-4, horizontalalignment='left', transform=axes.transAxes)
	fig_label = "RMSE"
	"""
	#
	if error_unit == "meV/atom":
		tmp_scale = 1000.0
		tmp_val1 = round(tmp_scale*train_err, 0)
		tmp_val2 = round(tmp_scale*test_err, 0)
	else:
		tmp_scale = 1.0
		tmp_val1 = round(tmp_scale*train_err, 3)
		tmp_val2 = round(tmp_scale*test_err, 3)
	axes.text(.01,.80, 'train %s = %.0f %s' % (fig_label, tmp_val1, error_unit), fontsize=fontsize-4, horizontalalignment='left', transform=axes.transAxes)
	axes.text(.01,.71, 'test %s = %.0f %s' % (fig_label, tmp_val2, error_unit), fontsize=fontsize-4, horizontalalignment='left', transform=axes.transAxes)
	"""
	#plt.legend(loc='upper left', frameon=False, fontsize='medium')
	#plt.text(0.5, 5.3, plot_label, fontsize=fontsize-4)
	#plt.text(1.5, 4.9, 'train %s = %.1f meV' % (fig_label, 1000*train_err), fontsize=fontsize-4)
	#plt.text(0.5, 4.5, 'test %s = %.1f meV' % (fig_label, 1000*test_err), fontsize=fontsize-4)
	# plt.xlim(xlim)
	# plt.ylim(ylim)
	# plt.xlim(xmin*1.02, xmax*.98)
	# plt.ylim(xmin*1.02, xmax*.98)
	#plt.annotate(('%s%s' %(r"R$^2$ = ", round(r2_score(y_test_pred, y_test), 3))), xy=(-7.65, -7.8), xytext=(-7.65, -7.8), fontsize=fontsize-4)
	plt.tick_params('both', length = 6, width = axis_width, which = 'major',right=True,top=True)
	fig.savefig(plot_name, dpi=600, bbox_inches='tight')
	plt.close()


def modified_pettifor(in_atom):
	
	mod_scale = {
			    "He": 1,
			    "Ne": 2,
			    "Ar": 3,
			    "Kr": 4,
			    "Xe": 5,
			    "Rn": 6,
			    "Fr": 7,
			    "Cs": 8,
			    "Rb": 9,
			    "K": 10,
			    "Na": 11,
			    "Li": 12,
			    "Ra": 13,
			    "Ba": 14,
			    "Sr": 15,
			    "Ca": 16,
			    "Eu": 17,
			    "Yb": 18,
			    "Lu": 19,
			    "Tm": 20,
			    "Y": 21,
			    "Er": 22,
			    "Ho": 23,
			    "Dy": 24,
			    "Tb": 25,
			    "Gd": 26,
			    "Sm": 27,
			    "Pm": 28,
			    "Nd": 29,
			    "Pr": 30,
			    "Ce": 31,
			    "La": 32,
			    "Ac": 33,
			    "Th": 34,
			    "Pa": 35,
			    "U": 36,
			    "Np": 37,
			    "Pu": 38,
			    "Am": 39,
			    "Cm": 40,
			    "Bk": 41,
			    "Cf": 42,
			    "Es": 43,
			    "Fm": 44,
			    "Md": 45,
			    "No": 46,
			    "Lr": 47,
			    "Sc": 48,
			    "Zr": 49,
			    "Hf": 50,
			    "Ti": 51,
			    "Ta": 52,
			    "Nb": 53,
			    "V": 54,
			    "Cr": 55,
			    "Mo": 56,
			    "W": 57,
			    "Re": 58,
			    "Tc": 59,
			    "Os": 60,
			    "Ru": 61,
			    "Ir": 62,
			    "Rh": 63,
			    "Pt": 64,
			    "Pd": 65,
			    "Au": 66,
			    "Ag": 67,
			    "Cu": 68,
			    "Ni": 69,
			    "Co": 70,
			    "Fe": 71,
			    "Mn": 72,
			    "Mg": 73,
			    "Zn": 74,
			    "Cd": 75,
			    "Hg": 76,
			    "Be": 77,
			    "Al": 78,
			    "Ga": 79,
			    "In": 80,
			    "Tl": 81,
			    "Pb": 82,
			    "Sn": 83,
			    "Ge": 84,
			    "Si": 85,
			    "B": 86,
			    "C": 87,
			    "N": 88,
			    "P": 89,
			    "As": 90,
			    "Sb": 91,
			    "Bi": 92,
			    "Po": 93,
			    "Te": 94,
			    "Se": 95,
			    "S": 96,
			    "O": 97,
			    "At": 98,
			    "I": 99,
			    "Br": 100,
			    "Cl": 101,
			    "F": 102,
			    "H": 103
			}

	return mod_scale[in_atom]/float(max(mod_scale.values()))
csv_file1='Ca1O1.csv'
csv_file2='Ag2O1.csv'
print('\n Using csv_file: %s for training and the csv file %s for testing\n\n'%(str(csv_file1),str(csv_file2)))
if __name__ == '__main__':
    main(csv_file1,csv_file2)
