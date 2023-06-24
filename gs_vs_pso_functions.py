#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 17 12:15:17 2021

@author: soroush m.veisi9687@gmail.com
"""
import numpy as np
import tkinter
import math
import scipy.optimize as optimization
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


#%%

def cal_qfactor(c):
    q = -np.pi * 1 / (np.log(10) * c * 3.2)
    return(q)
def cal_synthetic_value(pos, R, kink,M=3.0):
    error=0
    synth_val = np.zeros(shape=(len(R),))
    for i in range(len(R)):
        if (kink <= R[i]):
            Rs2=R[i]/ kink
            synth = pos[0] *M + pos[1]* math.log10(kink) \
                + pos[2]* math.log10(Rs2)*1 +pos[3]* R[i] + pos[4]
        else:
            synth = pos[0]* M + pos[1]* math.log10(R[i])+\
                pos[3]*R[i] + pos[4]
        synth_val[i] = synth 
    return(synth_val)
def cal_synthetic_value_4all(pos, R, M, kink= 90):
    error=0
    synth_val = np.zeros(shape=(len(R),))
    for i in range(len(R)):
        if (kink <= R[i]):
            Rs2=R[i]/ kink
            synth = pos[0] *M[i] + pos[1]* math.log10(kink) \
                + pos[2]* math.log10(Rs2)*1 +pos[3]* R[i] + pos[4]
        else:
            synth = pos[0]* M[i] + pos[1]* math.log10(R[i])+\
                pos[3]*R[i] + pos[4]
        synth_val[i] = synth 
    return(synth_val)
def cal_synth(pos,kink,log_A,R,M):
    na = np.shape(log_A)
    na = na[0]
    resi_out= np.zeros(na)
    error=0
    for i in range(na):
        if (kink <= R[i]):
            Rs2=R[i]/ kink
            synth = pos[0] *M[i] + pos[1]* math.log10(kink) \
                + pos[2]* math.log10(Rs2)*1 +pos[3]* R[i] + pos[4]
        else:
            synth = pos[0]* M[i] + pos[1]* math.log10(R[i])+\
                pos[3]*R[i] + pos[4]
        val = log_A[i]- synth 
        resi_out[i]= val
        error=error+ (val * val)
        
    # return resi_out, math.sqrt(error/na)
    return resi_out
#%%
# import scipy.optimize as optimization
# def ls_opt(x,y):
#     ls_out= optimization.curve_fit(scatter_rel, x, y)
#     slope_rad = ls_out[0][0]
#     intercept = ls_out[0][1]
#     return (slope_rad, intercept)
    
# def scatter_rel(params, x, y):
#     return (y+ np.dot(x,params))


def least_square_lfit(x, y):
    A = np.vstack([x[:, 0], np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    return(m ,c)
#%%

def find_slope_intercept(sol, log_amplitude, distance, magnitude, kink):
    ndata = len(sol)
    slope= np.zeros(ndata)
    intercept= np.zeros(ndata)
    def scatter_rel(params, x, y):
        return (y+ np.dot(x,params)) 
    for i in range(ndata):
        resi_out = cal_synth(sol[i,0:5], kink, log_amplitude\
                         ,distance, magnitude)
        ls_out= optimization.curve_fit(scatter_rel, distance, resi_out)
        slope[i]=ls_out[0][0]
        intercept[i]=ls_out[0][1]
        # slope[i], intercept[i] = ls_opt(distance, resi_out)
    return (slope, intercept)
#########################################################################
def find_slope_intercept_two_segment(sol, resi_out, log_amplitude, distance, magnitude, 
                                     kink):
    ndata = len(sol)
    slope= np.zeros(shape= (ndata, 2))
    intercept= np.zeros(shape= (ndata, 2))
    
    first_segment_ind = np.argwhere(distance <= kink)
    second_segment_ind = np.argwhere(distance > kink)
    
    log_amp_first_segment =  np.zeros(shape= (len(first_segment_ind),))
    distance_first_segment =  np.zeros(shape= (len(first_segment_ind),))
    magnitude_first_segment =  np.zeros(shape= (len(first_segment_ind),))
    
    log_amp_first_segment =  np.zeros(shape= (len(second_segment_ind),))
    distance_first_segment =  np.zeros(shape= (len(second_segment_ind),))
    magnitude_first_segment =  np.zeros(shape= (len(second_segment_ind),))
    
    
    log_amp_first_segment = log_amplitude[first_segment_ind]
    log_amp_second_segment = log_amplitude[second_segment_ind]
    
    distance_first_segment = distance[first_segment_ind]
    distance_second_segment = distance[second_segment_ind]
    
    magnitude_first_segment = magnitude[first_segment_ind]
    magnitude_second_segment = magnitude[second_segment_ind]
    
    
    for i in range(ndata):
        ydata = np.zeros(shape= (len(first_segment_ind),))
        ydata = resi_out[first_segment_ind, i]
        m, c= least_square_lfit(x= distance_first_segment, y= ydata)
        slope[i, 0] = m
        intercept[i, 0] = c
        ydata = np.zeros(shape= (len(second_segment_ind),))
        ydata = resi_out[second_segment_ind, i]
        m, c= least_square_lfit(x= distance_second_segment, y= ydata)
        slope[i, 1] = m
        intercept[i, 1] = c
        # slope[i], intercept[i] = ls_opt(distance, resi_out)
    poly_fit_ans = []
    for i in range(ndata):
        x = distance
        y = resi_out[:, i]
        poly_fit_ans.append(np.polyfit(x, y, 2))
    return (slope, intercept,poly_fit_ans,
            distance_first_segment, distance_second_segment)
     
#%%
def plot_slope_intercept(slope,intercept,good_solution,log_amplitude,\
                         distance,magnitude,name, kink):  
    ind_best = np.argwhere((good_solution[:,5] == np.min(good_solution[:,5])))
    resi_out = cal_synth(good_solution[ind_best[0][0],0:5], kink,\
                         log_amplitude,\
                         distance, magnitude)
    intercept_cond = 0.05
    slope_cond = 0.001
    gs= gridspec.GridSpec(3, 2)
    fig = plt.figure(figsize=(12,12))
    plt.subplots_adjust(hspace=0.3,wspace=0.3)
    ax = plt.subplot(gs[0,:])
    plt.scatter(distance, resi_out)
    ax.set_xlabel('Distance (km)')
    ax.set_ylabel('Residuals')

    ind_slope1= np.argwhere(abs(slope) < slope_cond)
    slope_bestd= slope[ind_slope1[:,0]]
    intercept_bestd= intercept[ind_slope1[:,0]]
    sol_bestd = good_solution[ind_slope1[:,0]]
    ind_intercept= np.argwhere(abs(intercept_bestd) < intercept_cond)
    intercept_best = intercept_bestd[ind_intercept[:,0]]
    slope_best = slope_bestd[ind_intercept[:,0]]
    sol_best = sol_bestd[ind_intercept[:,0]]
    if (len(slope_best) < 3000000):
        idum=0
        ndata=len(good_solution)
        for i in range(ndata):
            if (abs(slope[i]) <= slope_cond) and (abs(intercept[i]) <= intercept_cond):
                idum=idum+1
                plt.plot(distance, distance * slope[i] + intercept[i],'k')
    else:
        ind_of_min = np.argwhere(slope_best == np.min(slope_best))[0][0]
        plt.plot(distance, distance * slope[ind_of_min] + intercept[ind_of_min],'k')
        ind_of_max = np.argwhere(slope_best == np.max(slope_best))[0][0]
        plt.plot(distance, distance * slope[ind_of_max] + intercept[ind_of_max],'k')
        ind_of_min = np.argwhere(intercept_best == np.min(intercept_best))[0][0]
        plt.plot(distance, distance * slope[ind_of_min] + intercept[ind_of_min],'k')
        ind_of_max = np.argwhere(intercept_best == np.max(intercept_best))[0][0]
        plt.plot(distance, distance * slope[ind_of_max] + intercept[ind_of_max],'k')
    
    ax = plt.subplot(gs[1,0])
# plt.hist(slope, bins = 30, range=(-0.001, 0.001))
    # plt.hist(slope_best, bins=30, range=(-0.001, 0.001))
    N, bins, patches = ax.hist(slope, bins=100, color='navy')
    for c, p in zip(bins, patches):
        if (abs(c)< slope_cond):
            plt.setp(p, 'facecolor', 'aqua')
    ax.set_xlabel('Slope')
    ax.set_ylabel('Frequency')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))

    ax = plt.subplot(gs[1,1])
    N, bins, patches = ax.hist(intercept, bins=100, color='navy')
    for c, p in zip(bins, patches):
        if (abs(c)< intercept_cond):
            plt.setp(p, 'facecolor', 'aqua')
    ax.set_xlabel('Intercept')
    ax.set_ylabel('Frequency')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    
    ax= plt.subplot(gs[2,:])
    plt.hist2d(slope, intercept,bins=30, cmap='Blues')
    clb= plt.colorbar()
    clb.set_label('Frequency', labelpad=-30, y=1.07, rotation=0)
    ax.grid()
    ax.set_xlabel('Slope')
    ax.set_ylabel('Intercept')
    plt.ticklabel_format(axis="x", style="sci", scilimits=(0,0))
    plt.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    plt.show()
    fig.savefig(name, dpi=300, bbox_inches='tight')
    return(sol_best,slope_best,intercept_best)
def pick_for_plot(good_solution, all_solution, fig_plot_ind, ind_to_plot):
    par_to_plot = fig_plot_ind[ind_to_plot]
    if (par_to_plot == 'a_b1'):
        x_lab = 'b1'
        y_lab = 'a'
        par_x_all = np.array(all_solution[:, 1])
        par_x_good = np.array(good_solution[:, 1])
        par_y_all = np.array(all_solution[:, 0])
        par_y_good = np.array(good_solution[:, 0])
    elif (par_to_plot == 'a_b2'):
        x_lab = 'b2'
        y_lab = 'a'
        par_x_all = np.array(all_solution[:, 2])
        par_x_good = np.array(good_solution[:, 2])
        par_y_all = np.array(all_solution[:, 0])
        par_y_good = np.array(good_solution[:, 0])
    elif (par_to_plot == 'c_b1'):
        x_lab = 'b1'
        y_lab = 'c'
        par_x_all = np.array(all_solution[:, 1])
        par_x_good = np.array(good_solution[:, 1])
        par_y_all = np.array(all_solution[:, 3])
        par_y_good = np.array(good_solution[:, 3])
    elif (par_to_plot == 'c_b2'):
        x_lab = 'b2'
        y_lab = 'c'
        par_x_all = np.array(all_solution[:, 2])
        par_x_good = np.array(good_solution[:, 2])
        par_y_all = np.array(all_solution[:, 3])
        par_y_good = np.array(good_solution[:, 3])
    elif (par_to_plot == 'd_b1'):
        x_lab = 'b1'
        y_lab = 'd'
        par_x_all = np.array(all_solution[:, 1])
        par_x_good = np.array(good_solution[:, 1])
        par_y_all = np.array(all_solution[:, 4])
        par_y_good = np.array(good_solution[:, 4])
    elif (par_to_plot == 'd_b2'):
        x_lab = 'b2'
        y_lab = 'd'
        par_x_all = np.array(all_solution[:, 2])
        par_x_good = np.array(good_solution[:, 2])
        par_y_all = np.array(all_solution[:, 4])
        par_y_good = np.array(good_solution[:, 4])
    elif (par_to_plot == 'a_d'):
        x_lab = 'd'
        y_lab = 'a'
        par_x_all = np.array(all_solution[:, 4])
        par_x_good = np.array(good_solution[:, 4])
        par_y_all = np.array(all_solution[:, 0])
        par_y_good = np.array(good_solution[:, 0])
    elif (par_to_plot == 'b1_b2'):
        x_lab = 'b2'
        y_lab = 'b1'
        par_x_all = np.array(all_solution[:, 2])
        par_x_good = np.array(good_solution[:, 2])
        par_y_all = np.array(all_solution[:, 1])
        par_y_good = np.array(good_solution[:, 1])
    elif (par_to_plot == 'b2_b1'):
        x_lab = 'b1'
        y_lab = 'b2'
        par_x_all = np.array(all_solution[:, 1])
        par_x_good = np.array(good_solution[:, 1])
        par_y_all = np.array(all_solution[:, 2])
        par_y_good = np.array(good_solution[:, 2])
    elif (par_to_plot == 'c_d'):
        x_lab = 'd'
        y_lab = 'c'
        par_x_all = np.array(all_solution[:, 4])
        par_x_good = np.array(good_solution[:, 4])
        par_y_all = np.array(all_solution[:, 3])
        par_y_good = np.array(good_solution[:, 3])
    elif (par_to_plot == 'c_a'):
        x_lab = 'a'
        y_lab = 'c'
        par_x_all = np.array(all_solution[:, 0])
        par_x_good = np.array(good_solution[:, 0])
        par_y_all = np.array(all_solution[:, 3])
        par_y_good = np.array(good_solution[:, 3])
    return(par_x_all, par_x_good, par_y_all, par_y_good, x_lab, y_lab)
