# test_Counties.py
# Written by Ryan Murray 1/16/18

# The goal of this script is to build a function necessary to computing optimal voting districts which try to take into account geographic communities of interest. The companion paper proposed the following cost:
#   \sum_{g(i) = g(j)} \|\Gamma_i-\Gamma_j\|^2     (**)
#
# where here g is a function mapping districts to some geographic community and Gamma_i and Gamma_j are the ith and jth rows of the transportation plan Gamma.

#Included is a function which computes the linearized transport cost.


#############################################
#Importing necessary functions/libraries

from __future__ import division

import os
from subprocess import call
from glob import glob

import numpy as np
import pandas as pd
import geopandas as geo
import seaborn as sns
import multiprocessing as mp
import shapely
import pickle

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import transport_plan_functions as tpf
import make_maps as MM
from sklearn.metrics.pairwise import euclidean_distances



######################################################



#######################################################

#Inputs: Gamma - Current transportation plan
#        df    - Data frame.

#Output: Linearization of the cost (**) about the current transportation plan.

def linearized_County_Cost_Matrix(df):
    print 'Computing County Map'
    county_names = np.unique(df['COUNTY_NAM'])
    npcncts = len(df['COUNTY_NAM'])
    output = np.zeros((npcncts,npcncts))
    for county in county_names:
        #print county
        tmp = (df['COUNTY_NAM'] == county)
        N = np.sum(tmp)
        output += (np.diag(tmp) - np.outer(tmp,np.ones((1,npcncts)).dot(np.diag(tmp)))/N)/N
        #Note: the extra N here is so that each county counts the same towards the cost.

        #tmp_Gamma = np.diag(tmp).dot(Gamma)
        #s = np.average(tmp_Gamma,axis=0)
        #output += np.diag(tmp).dot(Gamma-s) #This uses a funny quirk, where subtracting a vector from a matrix will execute the operation to all the rows. May need some massaging to fix, but something like this should work.
    return output



#The following is pseudocode for solving:
#Given w fixed
#Find a transportation plan from v to w
#Which minimizes the standard transport distance plus the county distance
#I'm going to assume that the working directory is one of the data-files folders at the moment.

df = pd.read_pickle('../Data-Files/CO/precinct_data.p')




county_param = 1000 #10000 works well for CO. Normalization is non-obvious.
step_size = .5 #Step size for linearized county cost steps
reg=10 #.5?



# weight each precinct by its population share
n_districts = int(df.CD_2010.max()) + 1
n_precincts = len(df)
pcnct_loc = df[['INTPTLON10', 'INTPTLAT10']].values
pcnct_pop = np.maximum(df.POP_TOTAL.values, 20)
pcnct_wgt = pcnct_pop/pcnct_pop.sum()
I_wgt = pcnct_wgt
I_loc = pcnct_loc

F_wgt = np.ones((n_districts,))/n_districts
F_loc0 = np.zeros((n_districts, 2))
for i in range(n_districts):

	tmp = df[df['CD_2010'].values==i]

	# df['pop_area'] = df.POP_TOTAL/(df.area*1000)
	pop_argmax = tmp['POP_TOTAL'].argmax()

	F_loc0[i, 0] = tmp['INTPTLON10'].loc[pop_argmax]
	F_loc0[i, 1] = tmp['INTPTLAT10'].loc[pop_argmax]




num_it = 30 #Number of iterations for the county cost. Guess value: in tens
lloyd_steps = 30 #30
lineSearchN = 20 #20
lineSearchRange = 1
stepsize_vec = np.linspace(0,lineSearchRange,lineSearchN)
uinit = np.ones(len(I_wgt))/len(I_wgt)
F = F_loc0

dist_mat = euclidean_distances(I_loc, F)
Gamma,u,cost = tpf._computeSinkhorn(pcnct_wgt,F_wgt,dist_mat,reg,uinit)

lin_county_mat = linearized_County_Cost_Matrix(df)

print 'running Lloyd\'s Algorithm'
for i_step in range(1,lloyd_steps):
    #print i_step #debugging...
    
    #At each step find the the optimal transportation map Gamma given fixed F
    dist_mat = euclidean_distances(I_loc, F)

    for i in range(num_it):
        #print i
        lin_county = lin_county_mat.dot(Gamma)
        cost_mat = county_param*lin_county + dist_mat
        delta_Gamma,u,cost = tpf._computeSinkhorn(pcnct_wgt,F_wgt,cost_mat,reg,u)
        Gamma = Gamma + delta_Gamma*step_size

    tmp_dist = np.array([np.argmax(i) for i in Gamma])
    for i in range(n_districts):
        tmp_wgt = I_wgt*(tmp_dist ==i)
        F[i,0] = np.sum(I_loc[:,0]*tmp_wgt)/np.sum(tmp_wgt)
        F[i,1] = np.sum(I_loc[:,1]*tmp_wgt)/np.sum(tmp_wgt)

    #F[:,0] = Gamma.T.dot(I_loc[:,0]*I_wgt*7)/np.sum(Gamma,axis=0)
    #F[:,1] = Gamma.T.dot(I_loc[:,1]*I_wgt*7)/np.sum(Gamma,axis=0)


#    #Then step to the best F given fixed transportation map Gamma
#    Grad = tpf._transportGradient(pcnct_loc, pcnct_wgt, F, F_wgt, Gamma, dist_mat)
#
#    # find optimal step size (will step in direction of gradient)
#    costVec = np.zeros(lineSearchN)
#    for j in range(lineSearchN):
#        DistMat = euclidean_distances(I_loc,F-stepsize_vec[j]*Grad)
#    	cost = np.sum(DistMat*Gamma)
#        costVec[j] = cost 
#    
#    # find the optimal step size and adjust F accordingly
#    ind = np.argmin(costVec)
#    # print(np.min(costVec))
#    F -= stepsize_vec[ind]*Grad


df['district_final'] =	np.array([np.argmax(i) for i in Gamma])
plotter = MM.MapPlotter(df)

filename = 'before.png'
current_dists = df['CD_2010'].values.astype(int)

#Create palette
n_districts = len(df.CD_2010.unique())

# define the colormap
cmap = plt.cm.jet
cmaplist = [cmap(i) for i in range(cmap.N)]		
palette =[cmaplist[i] for i in range(0, cmap.N, int(cmap.N/n_districts))]	



colors = np.array([palette[i] for i in current_dists])
fig, ax = plt.subplots(1, 1, subplot_kw=dict(aspect='equal'))

patches = []
for poly in df.geometry.values:
    patches.append(Polygon(np.asarray(poly.exterior)))
patches = PatchCollection(patches)
patches.set_lw(.1)
patches.set_color(colors)

ax.add_collection(patches,autolim=True)
ax.autoscale_view()
		
ax.set_yticklabels([])
ax.set_xticklabels([])
fig.savefig(filename, bbox_inches='tight', dpi=100)
plt.close()			





#patches = plotter.plot_state(df, colors, ax, fig, filename)

filename = 'after.png'
colors = np.array([palette[i] for i in df['district_final']])
patches.set_color(colors)
# plot stars for office locations
for i in range(len(F)):
	ax.scatter(F[i, 0], F[i, 1], color='black', marker='*', s=30, alpha=1)	


fig.savefig(filename, bbox_inches='tight', dpi=300)


#Next step: take output and plot. Lines 414-415 in make_maps sets the colors for patches. plot_state() should make the state plot (see 399-402). A separate plot with the county boundaries would be sensible: could just reset the colors using a different field (namely 'COUNTY_NAM').
