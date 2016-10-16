from __future__ import division
import scipy as sp
import numpy as np

import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import pandas as pd

import geopandas as geo
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import shapely
import pickle
import time


def distance_metric(I_in, F_in, alphaW):
	'''
	Computes 'distance' between precincts (I_in) and offices (F_in)
	'''
	DistMat_travel       = metrics.pairwise.euclidean_distances( I_in[:,0:2], F_in[:,0:2] )
	DistMat_demographics = metrics.pairwise.euclidean_distances( np.atleast_2d( I_in[:,2]).T, np.atleast_2d( F_in[:,2] ).T)
	# DistMat_demographics *= DistMat_travel.mean()
	DistMat = DistMat_travel + alphaW*DistMat_demographics	
	return DistMat


def unpack_multipolygons(geos, districts):
	'''
	Takes a vector of polygons and/or multipolygons. Returns an array of 
	only polygons by taking multipolygons and unpacking them.
	'''
	geo_list = []
	district_list = []
	for g,d in zip(geos, districts):
		if type(g) == shapely.geometry.polygon.Polygon:
			geo_list.append(g)
			district_list.append(d)
		else:
			for sub_g in g:
				if type(sub_g) == shapely.geometry.polygon.Polygon:
					geo_list.append(sub_g)
					district_list.append(d)
	
	geos = np.array(geo_list)
	dists = np.array(district_list)
	return geos, dists


def computeTransportSinkhorn(distribS, distribT, M, reg, uin):
	'''
	Add some stuff here
	'''
	# init data
	Nini = len(distribS)
	Nfin = len(distribT)

	numItermax = 200
	cpt = 0

	# we assume that no distances are null except those of the diagonal of distances
	#u = np.ones(Nini)/Nini
	u = uin
	uprev=np.zeros(Nini)

	K = np.exp(-reg*M)	
	Kp = K*(1/np.atleast_2d(distribS).T)

	transp = K
	cpt = 0
	err=1

	while (err>1e-4 and cpt<numItermax):
		if np.logical_or(np.any(np.dot(K.T, u) == 0), np.isnan(np.sum(u))):
			# we have reached the machine precision
			# come back to previous solution and quit loop
			print('Infinity')
			if cpt!=0:
				u = uprev
			break
		uprev = u
		v = np.divide(distribT, np.dot(K.T,u))
		u = 1./np.dot(Kp, v)
		if cpt%3 == 0:
			# Currently checking for stopping every 5th iteration, may want to change
			transp = np.atleast_2d(u).T*K*v
			
			# transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
			err = np.linalg.norm((np.sum(transp,axis=0)-distribT))**2
		cpt = cpt +1


	Mout = np.atleast_2d(u).T*K*v
	temp = np.log(Mout)
	mask = np.isreal(np.log(Mout))
	temp[mask] = 0
	cost = np.sum(M*Mout) + 1.0/reg*np.sum( temp*Mout )
	return Mout,u,cost


def optimizeF(I, I_wgt, Tinit, Finit, F_wgt, DistMat):
	F_opt = Finit.copy()

	newtonSteps = 20
	for i in range(newtonSteps):
		F_opt -= transportGradient(Tinit, I, I_wgt, F_opt, F_wgt, DistMat)	

		# keep offices inside the state
		F_opt[:,2] = np.maximum(np.minimum(F_opt[:,2], 1), 0)
		F_opt[:,1] = np.maximum(np.minimum(F_opt[:,1], I[:,1].max()), I[:,1].min())
		F_opt[:,0] = np.maximum(np.minimum(F_opt[:,0], I[:,0].max()), I[:,0].min())

	return F_opt


def transportGradient(Mout, I, I_wgt, F, F_wgt, DistMat):
	'''
	computes gradient of transport problem in F
	'''
	# print Mout
	output = np.zeros(F.shape)	
	
	for j in range(len(F_wgt)):

		test = F[j,:] - I
		d = DistMat[:,j]
		dmask = d > .001			

		Mout_adjust = np.tile(Mout[dmask, j], (F.shape[1], 1)).T
		d_adjust = np.tile(d[dmask], (F.shape[1], 1)).T		
		# print Mout_adjust
		# print d_adjust
		output[j, :] = np.sum(Mout_adjust*test[dmask, :]/d_adjust, axis=0)
		
	return output


def interpolateTransportPlans(Tin,Tout,D):
	'''
	Takes two transport plans (MxN matrices), each representing a mapping from M
	fixed delta masses to N fixed delta masses. D represents the pairwise distance
	between the M masses. This function gradually changes Tin to Tout, by iteratively
	adding the closest point to each cluster. Implicitly this assumes that one
	label matches in Tin and Tout for each cluster.

	This need not assume that the N masses (district offices) are necessarily the
	same, but the regions should roughly coincide. This could be accomplished via
	an optimal transport problem between the two sets of district offices, and then
	reordering the transport plans accordingly. The M masses (precint locations)
	should be fixed for both.

	I'm going to assume that Tin and Tout are clean (ie take only the value 0,1),
	at least for the moment
	'''

	TCurrent = Tin.copy()

	while not np.array_equal(TCurrent, Tout):

		DTemp = np.zeros(D.shape)
		TMask = TCurrent*Tout
		for i in range(len(D)):
			j = np.nonzero(Tout[i, :])[0][0]
			if np.sum(TMask[i, :]) != 0:
				DTemp[i, :] = 0
			else:
				DTemp[i, :] = D[i, :]*TMask[:, j]

		minVal = np.min(DTemp[np.nonzero(DTemp)])
		indices = np.argwhere(DTemp == minVal)
		TCurrent[indices[0]] = Tout[indices[0]]


def gradientDescentOT(Iin, I_wgt, Fin, F_wgt, precinct_df, Tinit=None, reg=10, alphaW=0):
	'''
	'''
	n_districts = len(Fin)
	I = Iin
	F = Fin
	Nini = len(I_wgt)
	uinit = np.ones(Nini)/Nini

	#Initial optimization step, mostly to initialize u
	DistMat = distance_metric(I, F, alphaW)

	Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg, uinit)

	Mout_list = []
	F_list = []

	#For the moment I'm hard coding the number of Newton steps
	newtonSteps = 20
	lineSearchN = 10
	lineSearchRange = 2
	costVec = np.zeros(lineSearchN)
	stepsize_vec = np.linspace(0,lineSearchRange,lineSearchN)


	for i_step in range(1, newtonSteps):
		#Compute an optimal plan, given a certain I,F
		DistMat = distance_metric(I, F, alphaW)
		
		Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg,u)
		# print Mout
		precinct_df['district_iter' + str(i_step)] = np.array([np.argmax(i) for i in Mout])

		#Compute the Gradient in F
		Grad = transportGradient(Mout, I, I_wgt, F, F_wgt, DistMat)

		Mlist = []
		for j in range(lineSearchN):
			#Execute line search in the direction Grad
			DistMat = distance_metric(I, F - stepsize_vec[j]*Grad, alphaW)

			Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg, u)
			costVec[j] = cost
			Mlist.append(Mout)


		ind = np.argmin(costVec)
		print(np.min(costVec))
		F = F - stepsize_vec[ind]*Grad

		F_list.append(F)
		Mout_list.append(Mlist[ind])

	DistMat = distance_metric(I, F, alphaW)
	Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg,uinit)

	Mout_list.append(Mout)
	return Mout_list,cost,F_list

