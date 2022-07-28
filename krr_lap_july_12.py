import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
#
import re
import sys
import random 
import operator 
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

def mae(y_true, y_pred):
    return np.mean(abs(y_true-y_pred))

def me(y_true, y_pred):
    return np.mean(y_true-y_pred) 

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(np.mean((y_true-y_pred)**2))

def string_number_separator(string):
    import re
    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    res = temp.match(string).groups()
    return res

def list_of_atoms_and_numbers(string):
    atoms=[]
    atoms_number=[]
    str1=''
    while len(str1)<len(string):
        string1=str(string)[len(str1):]
        x,y=string_number_separator(string1)
        str1=str1+str(x)+str(y)
        atoms.append(str(x))
        atoms_number.append(int(y))
    return atoms, atoms_number

def input_column_creater(atom_list,prop):
    orbs=['1s','2s','2p','3s','3p','3d','4s','4p','4d','5s','5p','5d','6s','6p','tot']
    orbs_generated=[]
    for elements in atom_list:
        for i in range (len(orbs)):
            new_str=str(orbs[i]+str(elements))
            orbs_generated.append(new_str)
    for j in range (len(prop)):
        orbs_generated.append(str(prop[j]))
    return orbs_generated

def train_model(x_test, y_test, x_train, y_train, nfold, nthread, in_alpha=np.logspace(-15, 5, 21, base=2), in_gamma=np.logspace(-15, 3, 19, base=2), kernel=None, rseed=None):

    if rseed:
        random.seed(rseed)
        np.random.seed(rseed)

    neg_root_mean_squared_error = make_scorer(root_mean_squared_error, greater_is_better=False)

    clf = GridSearchCV(KernelRidge(kernel=kernel), cv=nfold, n_jobs=nthread, verbose=1, scoring=neg_root_mean_squared_error, param_grid={"alpha":in_alpha, "gamma": in_gamma})

    clf.fit(x_train, y_train)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)

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
    alpha = 0.2
    lw45 = 1.5
    axis_width=1.5
    figsize = (9.2,7)
    label_font = 24
    tick_font = 20
    leg_font = 20
    tick_len = 6
    test_marker = 'o'  #test_mark = '^'
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
    markersize = 80
    alpha = 0.6
    lw45 = 1.5
    axis_width=1.5

    train_err = mae(y1, x1)
    test_err = mae(y2, x2)
    print(("MAE TRAIN ERROR", train_err))
    print(("MAE TEST ERROR", test_err))

    fig = plt.figure(figsize=figsize)    
    plt.scatter(x1, y1, color=train_color, marker=train_marker, s=markersize, alpha=alpha, lw=0)
    plt.scatter(x2, y2, color=test_color, marker=test_marker, s=markersize, alpha=alpha, lw=0)
    plt.legend(['train', 'test'], fontsize=fontsize-4, frameon=False, loc='lower right')

    xmax = np.max(x1)
    xmin = np.min(x1)
    x45 = np.linspace(xmax*.98, xmin*1.02, num=100)
    y45 = x45
    plt.plot(x45, y45, color='black', lw=lw45, label='__nolegend__')

    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    axes = plt.gca()
    axes.set_title(axes.get_title()) #* 2)
    axes.set_xlabel(axes.get_xlabel(), size=fontsize) #* 0.5) # fontname="Times New Roman")
    axes.set_ylabel(axes.get_ylabel(), size=fontsize)

    fig_label = "RMSE"

    plt.tick_params('both', length = 6, width = axis_width, which = 'major',right=True,top=True)

    fig.savefig(plot_name, dpi=600, bbox_inches='tight')
    plt.show()

def string_number_separator(string):
    import re
    temp = re.compile("([a-zA-Z]+)([0-9]+)")
    res = temp.match(string).groups()
    return res

def list_of_atoms_and_numbers(string):
    atoms=[]
    atoms_number=[]
    str1=''
    while len(str1)<len(string):
        string1=str(string)[len(str1):]
        x,y=string_number_separator(string1)
        str1=str1+str(x)+str(y)
        atoms.append(str(x))
        atoms_number.append(int(y))
    return atoms, atoms_number

def input_column_creater(atom_list,prop):
    orbs=['1s','2s','2p','3s','3p','3d','4s','4p','4d','5s','5p','5d','6s','6p','tot']
    orbs_generated=[]
    for elements in atom_list:
        for i in range (len(orbs)):
            new_str=str(orbs[i]+str(elements))
            orbs_generated.append(new_str)
    for j in range (len(prop)):
        orbs_generated.append(str(prop[j]))
    return orbs_generated


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

def compound_pettifor(compound):
    cpd=compound.split('-')[0]
    in_atom,num_atom=list_of_atoms_and_numbers(cpd)
    if len(in_atom)==1:
        norm_pettifor=modified_pettifor(in_atom[0])
        return norm_pettifor
    else:
        norm_pettifor=0
        for i in range (len(in_atom)):
            tot=sum(num_atom)
            norm_pettifor+=(num_atom[i]/tot)*modified_pettifor(in_atom[i])
        return norm_pettifor

def get_random_sublist(df,indx, numb): # This method splits a given dataframe based on the column index
    #sub_list = random.sample(df[indx].tolist(), numb)
    list1=pd.unique(df[indx]).tolist()
    sub_list=random.sample(list1,numb)
    return df[df[indx].isin(sub_list)], df[~df[indx].isin(sub_list)]
'''
def get_random_sublist(df,indx, numb=1):
    sub_list = random.sample(df[indx].tolist(), numb)
    return df[df[indx].isin(sub_list)], df[~df[indx].isin(sub_list)]
'''
def minimum_k_point(df): # the following method gets a list that contains the k-index 'ik' value ranging from 0 to minimum index of k across all compounds in the dataframe
    list_k=[]
    list2=pd.unique(df['compound']).tolist()
    for entries in list2:
        mask = (df['compound']==str(entries))
        df1 = df.loc[mask]
        list3=pd.unique(df1['ik']).tolist()
        list_k.append(max(list3))
    b=min(list_k)
    lst=list(range(0,b+1))
    return lst
def list_k_point(df,numb): # this method utilizes the above list to get the projections corresponding to selected range of 'ik' values.
    lst=minimum_k_point(df)
    indx='ik'
    sub_list=random.sample(lst,numb)
    return df[df[indx].isin(sub_list)], df[~df[indx].isin(sub_list)]


def get_df(): # This trims the dataframe based on 'EF_PBE' values
    csv_file="jun_7_384_avg_cmpd_hse_one_shot.csv" # the datafile we are working on
    df1 = pd.read_csv(csv_file)
    mask = (df1['EF_PBE'] > -10.1) & (df1['EF_PBE'] <= 10.1)
    df = df1.loc[mask]
    df['atomtypes'] = [ re.findall('[A-Z][a-z]?', in_atom) for in_atom in df['compound'].tolist() ] 
    df['pet'] = [ modified_pettifor(in_atom[0])  for in_atom in df['atomtypes'].tolist() ] 
    #df['pet'] = [ compound_pettifor(compound) for compound in df['compound'] ]
    return df

def get_bg(inp_arr): #takes the numpy array of 'EF_HSE' (also for predicted values) and returns bandgap
    min_ar=np.min(inp_arr);max_ar=np.max(inp_arr)
    for entries in inp_arr:
        if entries <max_ar and entries>0:
            max_ar=entries
        if entries >min_ar and entries <=0:
            min_ar=entries
    #return min_ar,max_ar
    return max_ar-min_ar

def scatter_comp_nospin(x, y,plot_name):

    import os, sys
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    axis_width = 2
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['axes.linewidth'] = axis_width
    mpl.rcParams['axes.unicode_minus']=False

    fontsize = 20 #24
    alpha = 0.2
    lw45 = 1.5
    axis_width=1.5
    #figsize = (9.2,7)
    figsize=(9.6,7)
    label_font = 18 #24
    tick_font = 16 #20
    leg_font = 16 #20
    tick_len = 6
    test_marker = 'o'  #test_mark = '^'
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
    text_font = 16 #20
    axis_width = 2
    leg_fancy = False
    leg_frame = True
    leg_alpha = 1
    leg_loc = 'upper right'
    htp = 0.2
    leg_color = 'white'
    leg_edge_color = 'black'
    leg_shadow = False
    leg_lw = axis_width
    s = 50 #150
    pt_lw = 0
    decision_lw = axis_width
    decision_ls = '--'
    decision_color = 'black'
    markersize = 30 # 80
    alpha = 0.3
    lw45 = 1.5
    axis_width=1.5


    fig = plt.figure(figsize=figsize)
    plt.scatter(x, y, color='blue', marker=train_marker, s=markersize, alpha=alpha, lw=0)

    ylabel='E (in eV)'
    xlabel='k-index'
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    axes = plt.gca()
    axes.set_title(axes.get_title()) #* 2)
    axes.set_xlabel(axes.get_xlabel(), size=fontsize) #* 0.5) # fontname="Times New Roman")
    axes.set_ylabel(axes.get_ylabel(), size=fontsize)

    fig_label = "RMSE"

    plt.tick_params('both', length = 6, width = axis_width, which = 'major',right=True,top=True)

    fig.savefig(plot_name, dpi=80, bbox_inches='tight')
    plt.close()

def scatter_comp_spin(x1, y1,x2,y2,x3,y3,plot_name):

    import os, sys
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np

    axis_width = 2
    mpl.rcParams['mathtext.default'] = 'regular'
    mpl.rcParams['axes.linewidth'] = axis_width
    mpl.rcParams['axes.unicode_minus']=False

    fontsize = 24
    alpha = 0.2
    lw45 = 1.5
    axis_width=1.5
    #figsize = (9.2,7)
    figsize = (9.6,7)
    label_font = 18 #24
    tick_font = 16 #20
    leg_font = 16 #20
    tick_len = 6
    test_marker = 'o'  #test_mark = '^'
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
    text_font = 16 #20
    axis_width = 2
    leg_fancy = False
    leg_frame = True
    leg_alpha = 1
    leg_loc = 'upper right'
    htp = 0.2
    leg_color = 'white'
    leg_edge_color = 'black'
    leg_shadow = False
    leg_lw = axis_width
    s = 50 #150
    pt_lw = 0
    decision_lw = axis_width
    decision_ls = '--'
    decision_color = 'black'
    markersize = 20 #80
    alpha = 0.4
    lw45 = 1.5
    axis_width=1.5


    fig = plt.figure(figsize=figsize)
    plt.scatter(x1, y1, color='blue', marker='s', s=markersize, alpha=alpha, lw=0,label='HSE06') # was DW instead of HSE to indicate DW spin
    plt.scatter(x2, y2, color='red', marker='o', s=markersize, alpha=alpha, lw=0,label='ML $\Delta$-HSE06') # was UP instead of pred to indicate Up spin
    plt.scatter(x3, y3, color='black', marker='d', s=markersize, alpha=alpha, lw=0,label='PBE') # was UP instead of pred to indicate Up spin
    ylabel='E (in eV)'
    xlabel='k-index'

    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.legend()
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    axes = plt.gca()
    axes.set_title(axes.get_title()) #* 2)
    axes.set_xlabel(axes.get_xlabel(), size=fontsize) #* 0.5) # fontname="Times New Roman")
    axes.set_ylabel(axes.get_ylabel(), size=fontsize)

    fig_label = "RMSE"

    plt.tick_params('both', length = 6, width = axis_width, which = 'major',right=True,top=True)

    fig.savefig(plot_name, dpi=80, bbox_inches='tight')
    plt.close()

def df_split(inp_df):
    mask1 = (inp_df['EF_HSE'] <= 0.0) #
    mask2=(inp_df['EF_HSE'] > 0.0)  #  use & (condition) to include other condition as well
    df_val= inp_df.loc[mask1]
    df_con=inp_df.loc[mask2]
    return df_val,df_con

def lowest_non_negative(inp_array):
    a=np.sort(inp_array)
    for entries in a:
        if entries > 0.0:
            return entries
    return False

def highest_non_positive(inp_array):
    a=-np.sort(-inp_array)
    for entries in a:
        if entries <= 0.0:
            return entries
    return False


def band_plot(name,inp_df):
    df1=inp_df
    cmpd_list=df1['compound'].drop_duplicates().tolist()
    l1=[]
    b_err=[]
    for entries in cmpd_list:
        grouped = df1.groupby(df1.compound)
        df_new=grouped.get_group(entries)
        val,con=df_split(df_new)
        val=val.sort_values(by=['EF_HSE'], ascending=True)
        con=con.sort_values(by=['EF_HSE'], ascending=True)
        v_hse=val.eig_HSE.to_numpy()
        v_kpt=val.ik.to_numpy()
        c_kpt=con.ik.to_numpy()
        c_hse=con.eig_HSE.to_numpy()
        v_pred=val.eig_pred.to_numpy()
        c_pred=con.eig_pred.to_numpy()
        v_pbe=val.eig_PBE.to_numpy()
        c_pbe=con.eig_PBE.to_numpy()
        #print(entries,v_kpt,c_kpt,v_hse,c_hse,v_pred,c_pred,v_pbe,c_pbe)
        if len(v_kpt.tolist())!=0 and len(c_kpt.tolist())!=0:
            eig_hse=np.concatenate((v_hse,c_hse),axis=0)
            eig_pred=np.concatenate((v_pred,c_pred),axis=0)
            eig_pbe=np.concatenate((v_pbe,c_pbe),axis=0)
            mae_val=mae(v_hse, v_pred)
            mae_con=mae(c_hse,c_pred)
            mae_all=mae(eig_hse,eig_pred)
            hse_cbm=np.min(c_hse);hse_vbm=np.max(v_hse)
            me_val=me(v_hse, v_pred)
            me_con=me(c_hse,c_pred)
            me_all=me(eig_hse,eig_pred)
            pbe_cbm=np.min(c_pbe);pbe_vbm=np.max(v_pbe)
            pred_cbm=np.min(c_pred);pred_vbm=np.max(v_pred)
            if pred_cbm!=False and pred_vbm!=False:
                hse_bg=hse_cbm-hse_vbm
                pred_bg=pred_cbm-pred_vbm
                if pred_bg<=0:
                    lst=[]
                    c_list=[c_pred[0],c_pred[1],c_pred[2]]
                    v_list=[v_pred[0],v_pred[1],v_pred[2]]
                    for i in range (3):
                        for j in range (3):
                            diff=c_list[i]-v_list[j]
                            if diff>0:
                                 lst.append(diff)
                    pred_bg=min(lst)
                pbe_bg=pbe_cbm-pbe_vbm
                l1.append([str(entries),mae_val,mae_con,mae_all,me_val,me_con,me_all,pbe_bg,pred_bg,hse_bg,abs(pred_bg-hse_bg)])
                b_err.append(abs(pred_bg-hse_bg))
                kpt=np.concatenate((v_kpt,c_kpt),axis=0)
                scatter_comp_spin(kpt, eig_hse,kpt,eig_pred,kpt,eig_pbe,str(name)+str(entries)+'_HSE_bg='+str(hse_bg)+'_pred_bg='+str(pred_bg)+'.pdf')
    print(str(name)+'_bg_MAE is = ',np.mean(np.array(b_err)))
    heading=['compound','val_MAE','con_MAE','test_MAE','val_ME','con_ME','test_ME','pbe_bg','pred_bg','HSE_bg','abs_diff']
    df=pd.DataFrame(l1,columns=heading)
    df.to_csv(str(name)+'summary_bg.csv')    


def train_and_plot(name,train_df,test_df, nfold, nthread, ker, x_feats, y_feat, plot_gammas = False):
    
    number = random.randint(1, 1000)
    print("random seed", number)
    rseed = random.seed(number)

    #test_size = 0.20 
    # x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, shuffle=True)
    # x_data = df[keep_cols].to_numpy()
    # y_data = df[y_feat].to_numpy()

    x_train = train_df[x_feats].to_numpy()
    x_test = test_df[x_feats].to_numpy()

    y_train = train_df[y_feat].to_numpy()
    y_test = test_df[y_feat].to_numpy()

    print(np.shape(x_train))
    print(np.shape(x_test))
    
    #alphas=[0.000001] # for linear
    #gammas=[1.0] # for laplacian
    alphas = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0] #np.logspace(-15, 5, 21, base=2) #  # #
    gammas = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0] #np.linspace(0.001,0.1, 20) #np.logspace(-15, 5, 21, base=2) 

    res_dict = { idx: {'pred_test': [], 'pred_train': [], "best_params":{'alpha': 0, 'gamma': 0}} for idx in range(len(gammas)) }
    all_data_mae = []
    train_data_mae=[]
    for it, g in enumerate(gammas):
        clf, clf.best_params_, y_train, y_test, y_train_pred, y_test_pred = train_model(x_test, y_test, x_train, y_train, nfold, nthread, in_alpha=alphas, in_gamma=[g], kernel=ker, rseed=rseed)
        print(clf.best_params_)
        print(np.shape(y_train_pred), np.shape(y_test_pred))
        res_dict[it]['pred_train'].append( y_train_pred )
        res_dict[it]['pred_test'].append( y_test_pred )
        res_dict[it]['best_params']['alpha'] = clf.best_params_['alpha']
        res_dict[it]['best_params']['gamma'] = clf.best_params_['gamma']
        diff =  np.mean(abs(y_test - y_test_pred))
        train_diff =  np.mean(abs(y_train - y_train_pred))
        #print(it, 'train_MAE', train_diff)
        #print(it, "RMSE", root_mean_squared_error(y_test, y_test_pred))
        #print(it, "MAE", diff)
        train_data_mae.append(np.mean(train_diff))
        all_data_mae.append(np.mean(diff))

    min_index, min_value = min(enumerate(all_data_mae), key=operator.itemgetter(1))
    train_err=train_data_mae[min_index]

    tmp_figname =str(name)+'_'+'_'.join(["train_nsamples="+str(len(y_train)), "ker="+ker, "MAE="+str(min_value)])
    figname = tmp_figname+".pdf"
    xlabel = r"HSE06 eigenvalue_EF (eV)"
    ylabel = r"ML-pred eigenvalue_EF (eV)"
    error_unit = "meV"

    print('best parameters', "gamma:", res_dict[min_index]['best_params']['gamma'], "alpha:", res_dict[min_index]['best_params']['alpha'])
    scatter_comp(y_train, res_dict[min_index]['pred_train'],  y_test, res_dict[min_index]['pred_test'], error_unit, xlabel, ylabel, figname)

    if plot_gammas == True:
        tmp_gammas = [ res_dict[i]["best_params"]['gamma'] for i in range(len(gammas)) ] 
        tmp_alphas = [ res_dict[i]["best_params"]['gamma'] for i in range(len(gammas)) ] 
        #figsize = (9.2,7)
        #fig = plt.figure(figsize=figsize)    
        plt.scatter([np.log(i) for i in gammas], all_data_mae, color="blue", marker="o", s=80, alpha=0.6, lw=0)
    alpha=res_dict[min_index]['best_params']['alpha']
    gamma=res_dict[min_index]['best_params']['gamma']
    clf, clf.best_params_, y_train, y_test, y_train_pred, y_test_pred = train_model(x_test, y_test, x_train, y_train, nfold, nthread, in_alpha=[alpha], in_gamma=[gamma], kernel=ker, rseed=rseed)
    trn_df=train_df[['compound','ik','eig_PBE','eig_HSE','EF_HSE','EF_PBE']]
    trn_df['eig_pred']=y_train_pred
    tst_df=test_df[['compound','ik','eig_PBE','eig_HSE','EF_HSE','EF_PBE']]
    tst_df['eig_pred']=y_test_pred
    band_plot(str(name[:6])+'_train_',trn_df)
    band_plot(str(name[:6])+'_test_',tst_df)
    return train_err,min_value,alpha,gamma

df = get_df()

#inputs = ['pet', '1s', '2s', '2p', '3s', '3p', '3d', '4s', '4p', '4d', '5s', '5p', '5d', '6s', '6p', 'eig_PBE', 'EF_PBE']
inputs = ['pet', '1s_PBE', '2s_PBE', '2p_PBE', '3s_PBE', '3p_PBE', '3d_PBE', '4s_PBE', '4p_PBE', '4d_PBE', '5s_PBE', '5p_PBE', '5d_PBE', '6s_PBE', '6p_PBE', 'eig_PBE', 'EF_PBE','PBE_ACT','PBE_PD']
#inputs = ['1s', '2s', '2p', '3s', '3p', '3d', '4s', '4p', '4d', '5s', '5p', '5d', '6s', '6p', 'EF_PBE','PBE_ACT']
#target = 'eig_HSE'
#target='eigshift(eV)'
target='eig_HSE'
nfold = 5
nthread = 2
ker = 'laplacian'
trim_kpoints=True
#kpoint_numb=1
#cmpd_numb=50
#kpoint_numb_list=[5,10,20,50]
#cmpd_numb_list=[20,30,50]
kpoint_train_list=[1]
#kpoint_numb_list=[1,'all']
kpoint_numb_list=['all']
#cmpd_numb_list=[20,30,50]
cmpd_numb_list=[77]
list_err=[]
stats=[]
for cmpd_numb in cmpd_numb_list:
    for kpoint_train in kpoint_train_list:
        for kpoint_numb in kpoint_numb_list:
            l1=[]
            if trim_kpoints and str(kpoint_numb)!='all':
                for i in range(20):
                    train_df1, test_df1 = get_random_sublist(df,'compound',cmpd_numb)
                    train_df2,test_df2 = list_k_point(train_df1,kpoint_train)
                    train_df3,test_df3 = list_k_point(test_df1,kpoint_numb)
                    name='run_'+str(i+1)+'_'+str(cmpd_numb)+'_train_k_'+str(kpoint_train)+'_test_k_'+str(kpoint_numb)
                    train_mae,test_mae,alpha,gamma=train_and_plot(name,train_df2, train_df3, nfold, nthread, ker, inputs, target)
                    list_err.append([i+1,cmpd_numb,kpoint_train, kpoint_numb,train_mae,test_mae,alpha,gamma])
                    l1.append(test_mae)
            if trim_kpoints and str(kpoint_numb)=='all':
                for i in range(20):
                    train_df1, test_df1 = get_random_sublist(df,'compound',cmpd_numb)
                    train_df2,test_df2 = list_k_point(train_df1,kpoint_train)
                    name='run_'+str(i+1)+'_'+str(cmpd_numb)+'_train_k_'+str(kpoint_train)+'_test_k_'+str(kpoint_numb)
                    train_mae,test_mae,alpha,gamma=train_and_plot(name,train_df2, test_df1, nfold, nthread, ker, inputs, target)
                    list_err.append([i+1,cmpd_numb,kpoint_train, kpoint_numb,train_mae,test_mae,alpha,gamma])
                    l1.append(test_mae)
            avg=np.mean(np.array(l1))
            std=np.std(np.array(l1))
            stats.append([cmpd_numb,kpoint_train,kpoint_numb,avg,std])
heading=['runs','cmpd_num','kpoint_train','kpoint_num','train_MAE','test_MAE','alpha','gamma']
heading_stats=['cmpd_num','kpoint_train','kpoint_num','test_MAE','std']
dat_frame=pd.DataFrame(list_err,columns=heading)
dat_frame_stats=pd.DataFrame(stats,columns=heading_stats)
dat_frame.to_csv(str(ker)+'_train_test_summary.csv')
dat_frame_stats.to_csv('stats_'+str(ker)+'.csv')


