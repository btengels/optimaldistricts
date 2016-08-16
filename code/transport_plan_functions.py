from __future__ import division
import scipy as sp
import numpy as np


import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd


import geopandas as geo
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import shapely
import pickle
import time


def unpack_multipolygons(geos, districts, lakes):
	'''
	Takes a vector of polygons and/or multipolygons. Returns an array of 
	only polygons by taking multipolygons and unpacking them.
	'''
	geo_list = []
	district_list = []
	lake_list = []
	for g,d,lake in zip(geos, districts, lakes):
		if type(g) == shapely.geometry.polygon.Polygon:
			geo_list.append(g)
			district_list.append(d)
			lake_list.append(lake)
		else:
			for sub_g in g:
				if type(sub_g) == shapely.geometry.polygon.Polygon:
					geo_list.append(sub_g)
					district_list.append(d)
					lake_list.append(lake)
	
	geos = np.array(geo_list)
	dists = np.array(district_list)
	lakes = np.array(lake_list)
	return geos, dists, lakes


def plot_polygon_collection(ax, geoms, values=None, colormap='Set1', facecolor=None, edgecolor=None, alpha=0.5, linewidth=0, **kwargs):
	'''
	Plot a collection of Polygon geometries, much faster than GeoDataFrame.plot(...) and allows for updating on the fly.
	'''
	if values is not None:
		geoms, values = unpack_multipolygons(geoms, values)

	patches = []
	for poly in geoms:		
		a = np.asarray(poly.exterior)
		if poly.has_z:
			poly = shapely.geometry.Polygon(zip(*poly.exterior.xy))
		patches.append(Polygon(a))

	patches = PatchCollection(patches, facecolor=facecolor, linewidth=linewidth, edgecolor=edgecolor, alpha=alpha, **kwargs)

	if values is not None:
	    patches.set_array(values)
	    patches.set_cmap(colormap)

	ax.add_collection(patches, autolim=True)
	ax.autoscale_view()
	return patches


def computeTransportSinkhorn(distribS,distribT, M, reg,uin):
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
		if np.logical_or(np.any(np.dot(K.T,u)==0),np.isnan(np.sum(u))):
			# we have reached the machine precision
			# come back to previous solution and quit loop
			print 'Infinity'
			if cpt!=0:
				u = uprev
			break
		uprev = u
		v = np.divide(distribT,np.dot(K.T,u))
		u = 1./np.dot(Kp,v)
		if cpt%3==0:
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


# def Compute_Transport_Plan(F, F_wgt, I, I_wgt , reg=10):
# 	'''
# 	Computes the optimal transport map, and transport cost, between two discrete distributions in R^2.
# 	The first two columns of each matrix represent locations of points, the third represents a weight.
# 	Weights should be normalized so that both distributions have the same total mass
# 	(in other words the last column of I and F should have the same sum).

# 	INPUTS: I - an Mx2 matrix I, where M is the number of points in the first distribution,
# 			F - Nx2 matrix, where N is the number of points for the second distribution.

# 	Output: (fun,Mout) - a tuple,
# 			fun - the transport cost
# 			Mout - an MxN matrix P that represents the transport plan.
# 			Each entry (i,j) represents the amount of mass transfered from point i in M to point j in N.
# 			This, if coded properly, should be a very sparse matrix. A transport cost C is also
# 			outputed, which gives the total transport cost.
# 	'''
# 	DistMat = metrics.pairwise.euclidean_distances( I, F )
# 	Mout = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg)


# 	temp = np.log(Mout)
# 	mask = np.isreal(np.log(Mout))
# 	temp[mask] = 0
# 	cost = np.sum(DistMat*Mout) + 1.0/reg*np.sum( temp*Mout )

# 	return cost, Mout


def transportGradient(Mout,I,I_wgt,F,F_wgt,DistMat):
	'''
	computes gradient of transport problem in F
	'''
	# print Mout
	output = np.zeros(F.shape)	
	
	for j in range(len(F_wgt)):

		test = F[j,:] - I
		d = DistMat[:,j]
		dmask = d > .001			

		Mout_adjust = np.tile(Mout[dmask,j],(F.shape[1],1)).T
		d_adjust    = np.tile(d[dmask],(F.shape[1],1)).T		
		# print Mout_adjust
		# print d_adjust
		output[j,:] = np.sum( Mout_adjust*test[dmask,:]/d_adjust, axis=0 )
		
	return output




def optimizeF(I,I_wgt,Tinit,Finit,F_wgt,DistMat):
	output = Finit.copy()

	newtonSteps = 20
	for i in range(newtonSteps):
		output -= transportGradient(Tinit,I,I_wgt,output,F_wgt,DistMat)

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

	while not np.array_equal(TCurrent,Tout):

		DTemp = np.zeros(D.shape)
		TMask = TCurrent*Tout
		for i in range(len(D)):
			j = np.nonzero(Tout[i,:])[0][0]
			if np.sum(TMask[i,:]) != 0:
				DTemp[i,:] = 0
			else:
				DTemp[i,:] = D[i,:]*TMask[:,j]

		minVal = np.min(DTemp[np.nonzero(DTemp)])
		indices = np.argwhere(DTemp == minVal)
		TCurrent[indices[0]] = Tout[indices[0]]



def gradientDescentOptimalTransport(Iin,I_wgt,Fin,F_wgt,precinct_data, Tinit=None, reg=10, alphaW=0):
	'''
	'''
	n_districts = len(Fin)
	I = Iin
	F = Fin
	Nini = len(I_wgt)
	uinit = np.ones(Nini)/Nini

	#Initial optimization step, mostly to initialize u
	DistMat_travel       = metrics.pairwise.euclidean_distances( Iin[:,0:2], Fin[:,0:2] )
	DistMat_demographics = metrics.pairwise.euclidean_distances( np.atleast_2d( Iin[:,2]).T, np.atleast_2d( Fin[:,2] ).T)
	DistMat_demographics *= DistMat_travel.mean()
	DistMat = (1-alphaW)*DistMat_travel + alphaW*DistMat_demographics
	

	Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg,uinit)

	Mout_list = []
	F_list = []

	#For the moment I'm hard coding the number of Newton steps
	newtonSteps = 20
	lineSearchN = 5
	lineSearchRange = 2
	costVec = np.zeros(lineSearchN)
	stepsize_vec = np.linspace(0,lineSearchRange,lineSearchN)


	for i_step in range(1,newtonSteps):
		#Compute an optimal plan, given a certain I,F
		DistMat_travel       = metrics.pairwise.euclidean_distances( I[:,0:2], F[:,0:2] )
		DistMat_demographics = metrics.pairwise.euclidean_distances( np.atleast_2d( I[:,2]).T, np.atleast_2d( F[:,2] ).T)
		DistMat_demographics *= DistMat_travel.mean()
		DistMat = (1-alphaW)*DistMat_travel + alphaW*DistMat_demographics
		
		Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg,u)
		# print Mout
		precinct_data['district_iter'+str(i_step)] = np.array( [np.argmax(i) for i in Mout])


		#Compute the Gradient in F
		
		Grad = transportGradient(Mout,I,I_wgt,F,F_wgt,DistMat)

		Mlist = []
		for j in range(lineSearchN):
			#Exectue line search in the direction Grad
			# print Grad
			# print stepsize_vec[j]*Grad[:,0:2]
			# print F[:,0:2] - stepsize_vec[j]*Grad[:,0:2]

			DistMat_travel = metrics.pairwise.euclidean_distances( I[:,0:2], F[:,0:2] - stepsize_vec[j]*Grad[:,0:2] )
			
			# DistMat_demographics = metrics.pairwise.euclidean_distances( np.I, F - stepsize_vec[j]*Grad )
			DistMat_demographics = metrics.pairwise.euclidean_distances( np.atleast_2d( I[:,2]).T, np.atleast_2d( F[:,2] - stepsize_vec[j]*Grad[:,2]).T)
			DistMat_demographics *= DistMat_travel.mean()
			DistMat = (1-alphaW)*DistMat_travel + alphaW*DistMat_demographics

			Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg, u)
			costVec[j] = cost
			Mlist.append(Mout)


		ind = np.argmin(costVec)
		print np.min(costVec)
		F = F - stepsize_vec[ind]*Grad

		F_list.append(F)
		Mout_list.append(Mlist[ind])


	DistMat_travel       = metrics.pairwise.euclidean_distances( I[:,0:2], F[:,0:2] )
	DistMat_demographics = metrics.pairwise.euclidean_distances( np.atleast_2d( I[:,2]).T, np.atleast_2d( F[:,2] ).T)
	DistMat_demographics *= DistMat_travel.mean()
	DistMat = (1-alphaW)*DistMat_travel + alphaW*DistMat_demographics
	Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg,uinit)

	Mout_list.append(Mout)
	return Mout_list,cost,F_list


# ------------------------------------------------------------------------------
# make scatter plot and map
# ------------------------------------------------------------------------------
if __name__ == '__main__':
	
	make_scatterfig(I, F_opt, precinct_data.district.values, filename='figures/CO_dots.pdf')
	make_map(census_geo, precinct_data, F_opt, filename='figures/CO_map_Test.pdf')
