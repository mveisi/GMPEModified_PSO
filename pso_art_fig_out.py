#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Aug 19 20:25:45 2021

@author: soroush m.veisi9687@gmail.com
"""
import os
import tkinter
import numpy as np
import matplotlib.pyplot as plt
import gs_vs_pso_functions as gpf
from sklearn.metrics import silhouette_samples, silhouette_score
from matplotlib import cm
import matplotlib

cwd_path = os.getcwd()
with open("inputarguments", "r") as inparg:
    lines = inparg.readlines()
    for i, el in enumerate(lines):
        el = el.split()[0]
        if (i == 0):
            input_file = el
        elif (i == 1):
            if (el == 'True'):
                syn_run = True 
            else:
                syn_run = False
        elif (i == 2):
            syn_error = float(el)
        elif (i == 3):
            if (el == "True"):
                syn_run = False 
            else:
                syn_run = True
        elif (i == 4):
            kink = float(el)
        elif (i == 5):
            r_ref = float(el)
        elif (i == 6):
            n_ans = float(el)
        else:
            output_folder = el + '/'
            
#%%     






show_kmean = False #you can see the kmean cluster centers by change this to True

#following is part of code which i used for different synthetic run, 
#you can save different synthetic output of MATLAB code and run it here
if (syn_run):
    percent_to_consider = 2.7 #percent to consider for coloring the particle
                              #the particle with minimum cost function + 
                              #this_percent * minimum cost function will
                              #be colored and considered for later computation
                              #you can specify this according to your dataset
    
    output_figs = os.path.join(cwd_path, output_folder) 
    out_org = np.loadtxt(output_folder+ 'all_particle_for_save_syn.dat') #path of 
                                #all_particle_for_save.dat outputed from MATLAB
    pso_ls_merge = np.loadtxt(output_folder+'t3_ls_merge.dat') #path of 
                                #t3_ls_merge.dat outputed from MATLAB
    data = np.loadtxt(output_folder+ 'data_syn.dat') #path of 
                                #data_syn.dat outputed from MATLAB 
                                #for synthetic data
    thresh = (np.min(pso_ls_merge[:, 5]) + 
               np.min(pso_ls_merge[:, 5]) * (percent_to_consider / 100))
    name_of_out_fig='pso_syn_err_'+str(syn_error)+'_pso_ls'
    output_figs = os.path.join(output_figs, name_of_out_fig)
    print("Starting to create figures for synthetic run")
    if (syn_error >= 70):
        percent_to_consider = 1.0
        
else:
    percent_to_consider = 2.7
    output_figs = os.path.join(cwd_path, output_folder)
    out_org = np.loadtxt(output_folder+ 'all_particle_for_save.dat') #path of 
                                #all_particle_for_save.dat outputed from MATLAB
    pso_ls_merge = np.loadtxt(output_folder+'t3_ls_merge.dat') #path of 
                                #t3_ls_merge.dat outputed from MATLAB
    data = np.loadtxt(input_file) #path of 
                                #data_syn.dat outputed from MATLAB 
                                #for synthetic data
    thresh = (np.min(pso_ls_merge[:, 5]) + 
               np.min(pso_ls_merge[:, 5]) * (percent_to_consider / 100))
    print("Starting to create figures for real dataset run")
    name_of_out_fig='modified_pso_real'
    output_figs = os.path.join(output_figs, name_of_out_fig)
    
    
    
if (os.path.isdir(output_figs)):
    pass
else:
    os.mkdir(output_figs)
data_little_ref_ind=np.argwhere(data[:, 0] < r_ref)
data_little_ref = np.array(data[data_little_ref_ind[:, 0],:])






#%%
global distance
global magnitude
global log_amplitude
distance = np.array(data_little_ref[:, 0])
magnitude = np.array(data_little_ref[:, 1])
log_amplitude = np.array(data_little_ref[:, 2])


#%% finding colored particles
good_solution_ind = np.argwhere(out_org[:, 5] <= thresh)
good_solution = np.array(out_org[good_solution_ind[:, 0],:])

if (kink >= r_ref):
    good_solution[:, 2] = 0.0
    pso_ls_merge[:, 2] = 0.0

#%%

first_run = 1
if (first_run == 1):
    slope, intercept = gpf.find_slope_intercept(good_solution, \
                            log_amplitude, distance, magnitude, kink= kink)
    np.savetxt('slope_ftry_db.dat', slope)
    np.savetxt('intercept_ftry_db.dat', intercept)
else:
    slope=np.loadtxt('slope_ftry_db.dat')
    intercept=np.loadtxt('intercept_ftry_db.dat')
    

#%% finding confidence interval particles
sol_best, slope_best,intercept_best= gpf.plot_slope_intercept(slope,
                            intercept,good_solution,log_amplitude,
                    distance,magnitude,output_figs+'/fig_out_fin_art.png', kink = kink)

    
#%% Using dbscan algorithm for clustering
# from sklearn.cluster import DBSCAN
# from sklearn.neighbors import NearestNeighbors
# import matplotlib.pyplot as plt
# from sklearn.metrics import  silhouette_score
# pso_out = np.loadtxt('output_pso.dat')
# X=np.vstack((sol_best[:,0:5],pso_out[:,0:5]))
# neigh = NearestNeighbors(n_neighbors=5)
# nbrs = neigh.fit(X)
# distances, indices = nbrs.kneighbors(X)
# distances2 = np.sort(distances, axis=0)
# plt.figure(figsize=(12,12))
# plt.plot(distances2[:,4])
# plt.savefig('distance2', dpi=300, bbox_inches='tight')
# epsilon=0.03
# n_sample_limit= 2
# idum=-1
# epsilond= np.arange(0.01, 0.09, 0.01)
# n_sample_limitd=np.arange(5,9)
# db_scan_info = np.zeros([len(epsilond)*len(n_sample_limitd),3])
# # for epsilon in np.arange(0.01, 0.06, 0.01):
# #     for n_sample_limit in np.arange(2,7):
# #         idum=idum+1
# #         db = DBSCAN(eps=epsilon, min_samples=n_sample_limit).fit(X)
# #         core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# #         core_samples_mask[db.core_sample_indices_] = True
# #         labels = db.labels_
# #         n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# #         n_noise_ = list(labels).count(-1)
# #         sil_score = silhouette_score(X, labels, metric='euclidean')
# #         print('Estimated number of clusters: %d' % n_clusters_)
# #         print('Estimated number of noise points: %d' % n_noise_)
# #         print('The Silhoutte score is: %f' % sil_score)
# #         db_scan_info[idum,0]=epsilon
# #         db_scan_info[idum,1]=n_sample_limit
# #         db_scan_info[idum,2]=sil_score

# epsilon=0.1
# n_sample_limit=5
# db = DBSCAN(eps=epsilon, min_samples=n_sample_limit).fit(X)
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
# sil_score = silhouette_score(X, labels, metric='euclidean')
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# print('The Silhoutte score is: %f' % sil_score)


# for i in np.arange(len(X)-1, len(X)-len(pso_out)-1, -1):
#     name = "plot_scatter_for_" + str(i)
#     label_ref = labels[i]
#     ind = np.argwhere(labels == label_ref)
#     plt.figure(figsize=(12,12))
#     plt.scatter(X[ind,0],X[ind,1])
#     plt.savefig(name, dpi=300, bbox_inches='tight')
    
    


    
#%%
if (show_kmean):
    from sklearn.cluster import KMeans

    distortions = []
    score_silhouette = []
    core_kmean = []
    cluster_labels = np.zeros(shape= (len(sol_best), len(range(1, 10))))
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k, random_state=10)
        kmeanModel.fit(sol_best[:,0:5])
        distortions.append(kmeanModel.inertia_)
        core_kmean.append(kmeanModel.cluster_centers_)
        cluster_labels[:, k-1] = kmeanModel.labels_
        cl_label_4_sil = kmeanModel.labels_
        if (k > 1):
            score = silhouette_score(sol_best[:,0:5], cl_label_4_sil)
            score_silhouette.append(score)
        
    fig, axs = plt.subplots(nrows=2, ncols=1,figsize=(16,8), sharex= True)
    # plt.tight_layout()
    axs[0].plot(K, distortions, 'bx-', color = 'blue')
    axs[0].set_ylabel('Distortion', size = 20)
    axs[0].set_title('The Elbow Method showing the optimal k', size = 20)
    axs[1].plot(K[1:], score_silhouette, 'bx-', color = 'red')
    axs[1].set_xlabel('Number of Kmean clusters (k)', size = 20)
    axs[1].set_ylabel('Silhouette Score', size = 20)
    axs[1].set_title('The Silhouette Score showing the optimal k', size = 20)


    plt.show()
    fig.savefig(output_figs+'/elbow_method_for_gs_stry'+'.png', 
                dpi=300, bbox_inches='tight')
    n_cluster_final = 3
    sol_kmean = core_kmean[n_cluster_final - 1]
#%%



if ((syn_run) or (kink > 1000)):
    fig_plot_ind = ['a_b1', 'c_b1',
                'd_b1', 'a_d',
                'c_d', 'c_a']
                
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(16, 16)
                        ,facecolor='#FAEBD7')
else:
    fig_plot_ind = ['a_b1', 'a_b2',
                    'c_b1', 'c_b2', 
                    'd_b1', 'd_b2',
                    'a_d', 'b1_b2',
                    'c_d', 'c_a']
    fig, axs = plt.subplots(nrows=5, ncols=2, figsize=(16, 16))

good_solution = np.array(good_solution)
resi_std = good_solution[:, 5]

sort_ind = sorted(range(len(resi_std)), key=lambda k: resi_std[k])
good_sol_sorted = []
for i in reversed(range(len(good_solution))):
    good_sol_sorted.append(good_solution[sort_ind[i]])
good_sol_sorted = np.array(good_sol_sorted)
good_solution = []
good_solution = good_sol_sorted.copy()
resi_std = np.array(good_solution[:, 5])


if (show_kmean == False):
    sol_pso_ls = pso_ls_merge[:, :-1]
    sol_kmean = sol_pso_ls.copy()


solution = np.zeros(shape=(len(sol_kmean), 6))


colors_table = ['red', 'lightgreen', 'orange', 'yellow', 'wheat', 'lavender', 
                'khaki']
best_solution = np.array(sol_best)
resi_std_best = best_solution[:, 5]

sort_ind = sorted(range(len(resi_std_best)), key=lambda k: resi_std_best[k])
best_solution_sorted = []
for i in reversed(range(len(best_solution))):
    best_solution_sorted.append(best_solution[sort_ind[i]])
best_solution_sorted = np.array(best_solution_sorted)
best_solution = []
best_solution = best_solution_sorted.copy()
resi_std_best = np.array(best_solution[:, 5])


# for i in range(len(sol_best[:5])):
for i in range(len(sol_kmean)):
    resi = gpf.cal_synth(sol_kmean[i, :], kink, log_amplitude, 
                          distance, magnitude)
    solution[i, :5] = sol_kmean[i, :]
    solution[i, 5]= np.std(resi) 

idum = -1
for i in range(len(axs)):
    for j in range(2):
        idum+= 1
        (par_x_all, par_x_good, par_y_all, par_y_good, x_lab, y_lab) = \
            gpf.pick_for_plot(good_solution, out_org, fig_plot_ind,
                                        ind_to_plot= idum)
        axs[i, j].scatter(par_x_all, par_y_all, s= 10, color = 'gray')
        sc= axs[i, j].scatter(par_x_good, par_y_good, s= 10, c = resi_std, 
                          cmap = 'rainbow')
        
        (par_x_all, par_x_good, par_y_all, par_y_good, x_lab, y_lab) = \
            gpf.pick_for_plot(best_solution, out_org, fig_plot_ind,
                                        ind_to_plot= idum)
        axs[i, j].scatter(par_x_good, par_y_good, s= 10, color = 'black', 
                          alpha = 0.1)
        
        (par_x_all, par_x_good, par_y_all, par_y_good, x_lab, y_lab) = \
            gpf.pick_for_plot(solution, out_org, fig_plot_ind,
                                        ind_to_plot= idum)
        for k in range(len(par_x_good)):
            if ( k != len(par_x_good) - 1):
                axs[i, j].scatter(par_x_good[k], par_y_good[k], s= 300, marker = '*', 
                              c = colors_table[k])
            else:
                axs[i, j].scatter(par_x_good[k], par_y_good[k], s= 100, marker = 's', 
                              c = 'brown')
    
        axs[i, j].set_xlabel(x_lab, size = 18)
        axs[i, j].set_ylabel(y_lab, size = 18)
plt.tight_layout()
plt.colorbar(sc, ax=axs[:, :],shrink=0.24).set_label(label='STD of residuals',
                                                     size=15)
plt.savefig(output_figs+'/all_fig_tradeoff.png', dpi=300, bbox_inches='tight')

#%%

fig2, axs2 = plt.subplots(nrows=2, ncols=1,figsize=(16, 16) )
if ((syn_run) or (kink > 1000)):
    fig_plot_ind = ['c_b1', 'd_b1']
else:
    fig_plot_ind = ['b2_b1', 'd_b1']
idum = -1
for i in range(len(axs2)):
    idum+= 1
    (par_x_all, par_x_good, par_y_all, par_y_good, x_lab, y_lab) = \
            gpf.pick_for_plot(good_solution, out_org, fig_plot_ind,
                                        ind_to_plot= idum)
    axs2[i].scatter(par_x_all, par_y_all, s= 10, color = 'gray')
    sc= axs2[i].scatter(par_x_good, par_y_good, s= 10, c = resi_std, 
                      cmap = 'rainbow')
   
    (par_x_all, par_x_good, par_y_all, par_y_good, x_lab, y_lab) = \
        gpf.pick_for_plot(best_solution, out_org, fig_plot_ind,
                                    ind_to_plot= idum)
    axs2[i].scatter(par_x_good, par_y_good, s= 10, color = 'black', alpha = 0.1)
    
    
    (par_x_all, par_x_good, par_y_all, par_y_good, x_lab, y_lab) = \
        gpf.pick_for_plot(solution, out_org, fig_plot_ind,
                                    ind_to_plot= idum)
    for k in range(len(par_x_good)):
        if ( k != len(par_x_good) - 1):
            axs2[i].scatter(par_x_good[k], par_y_good[k], s= 300, marker = '*', 
                          c = colors_table[k])
        else:
            axs2[i].scatter(par_x_good[k], par_y_good[k], s= 100, marker = 's', 
                          c = 'brown')
    axs2[i].set_xlabel(x_lab, size = 18)
    axs2[i].set_ylabel(y_lab, size = 18)
plt.tight_layout()
plt.colorbar(sc, ax=axs2[:],shrink=0.24).set_label(label='STD of residuals',
                                                     size=15)
plt.savefig(output_figs+'/two_fig_tradeoff', dpi=300, bbox_inches='tight')
#%% fig12 plot
# if (syn_run):
#     fig_12 = plt.figure(figsize=(16, 10))
#     distance_syn = np.linspace(2, 70, 69)
#     sort_ind = sorted(range(len(distance)), key=lambda k: distance[k])
#     dis_sort = []
#     mag_sort = []
#     log_amp_sort = []
#     for i in range(len(sort_ind)):
#         dis_sort.append(distance[sort_ind[i]])
#         mag_sort.append(magnitude[sort_ind[i]])
#         log_amp_sort.append(log_amplitude[sort_ind[i]])
    
#     plt.scatter(dis_sort, log_amp_sort, s= 40, c= 'gray')
#     for i in range(len(solution)):
#         synth_val = gpf.cal_synthetic_value_4all(solution[i, :5], dis_sort,
#                                             kink = kink ,M= mag_sort)
#         plt.plot(dis_sort, synth_val, lw = 2.0, color = colors_table[i])
#     plt.title('Possible Solutions', size = 20)
#     plt.xlabel('Distance (km)', size = 20)
#     plt.ylabel('log (A)', size = 20)
#     plt.grid()
#     plt.savefig(output_figs+'att_curve', dpi=300, bbox_inches='tight')
# else:
condition1 = magnitude >= 2.8
condition2 = magnitude <= 3.2

# get indices where both conditions are met
indices_12 = np.argwhere(np.logical_and(condition1, condition2))

magnitude_12 = magnitude[indices_12]
log_amplitude_12 = log_amplitude[indices_12]
distance_12 = distance[indices_12]

fig_12 = plt.figure(figsize=(16, 10))

distance_syn = np.linspace(2, 70, 69)

plt.scatter(distance_12, log_amplitude_12, s= 40, c= 'gray')
for i in range(len(solution)):
    synth_val = gpf.cal_synthetic_value(solution[i, :5], distance_syn,
                                        kink = kink ,M=3.0)
    if (i == len(solution) - 1):
        plt.plot(distance_syn, synth_val, lw = 2.0, color = 'brown')
    else:
        plt.plot(distance_syn, synth_val, lw = 2.0, color = colors_table[i])
plt.title('Possible Solutions for Mw = 2.8-3.2', size = 20)
plt.xlabel('Distance (km)', size = 20)
plt.ylabel('log (A)', size = 20)
plt.grid()
plt.savefig(output_figs+'/att_curve', dpi=300, bbox_inches='tight')   
        
  
    
#%%
q_factor = []
for i in range(len(solution)):
    q_factor.append(gpf.cal_qfactor(solution[i, 3]))
## error of clusters
# cluster_labels_chosen = cluster_labels[:, n_cluster_final]
# solution_error = np.zeros(shape=(len(solution), 5))

# for i in range(len(solution)):
#     ind = np.argwhere(cluster_labels_chosen == i)
#     for j in range(5):
#         solution_error[i, j] = np.std(sol_best[ind, j])    