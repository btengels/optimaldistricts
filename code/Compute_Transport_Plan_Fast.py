from __future__ import division
import scipy as sp
import numpy as np
from scipy.optimize import linprog
import scipy.optimize as opt
import sklearn.metrics as metrics

import matplotlib.pyplot as plt

from images2gif import writeGif
from PIL import Image


import seaborn as sb
import pandas as pd
import cvxopt as cvx
from cvxopt.solvers import lp

import geopandas as geo
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import shapely
import pickle
import time

cvx.solvers.options['show_progress'] = False




def plot_polygon_collection(ax, geoms, values=None, colormap='Set1',facecolor=None, edgecolor=None, alpha=0.5, linewidth=0, **kwargs):
	'''
	Plot a collection of Polygon geometries, much faster than GeoDataFrame.plot(...) and allows for updating on the fly.
	'''
	patches = []
	mask = np.array([type(i)==shapely.geometry.polygon.Polygon for i in geoms])

	for poly_true, poly in zip(mask,geoms):

		if poly_true:
			a = np.asarray(poly.exterior)
			if poly.has_z:
				poly = shapely.geometry.Polygon(zip(*poly.exterior.xy))

			patches.append(Polygon(a))

		else:
			a = np.asarray(poly[0].exterior)
			if poly.has_z:
				poly = shapely.geometry.Polygon(zip(*poly[0].exterior.xy))

			patches.append(Polygon(a))


	patches = PatchCollection(patches, facecolor=facecolor, linewidth=linewidth, edgecolor=edgecolor, alpha=alpha, **kwargs)

	if values is not None:
	    patches.set_array(values)
	    patches.set_cmap(colormap)

	ax.add_collection(patches, autolim=True)
	ax.autoscale_view()
	return patches



def computeTransportSinkhorn(distribS,distribT, M, reg,uin):
	# Sinkhorn algorithm from Cuturi 2013.
	'''
	M is a matrix of pairwise distances
	distribS is a distribution of initial weights
	distribT is a distribution of final weights
	reg is a regularization parameter
	uin is an initial guess for the Lagrange multiplier u
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
	Kp = np.dot(np.diag(1/distribS),K)
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
			transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
			err = np.linalg.norm((np.sum(transp,axis=0)-distribT))**2
		cpt = cpt +1

#
	Mout = np.dot(np.diag(u),np.dot(K,np.diag(v)))
	temp = np.log(Mout)
	mask = np.isreal(np.log(Mout))
	temp[mask] = 0
	cost = np.sum(M*Mout) + 1.0/reg*np.sum( temp*Mout )
	return Mout,u,cost



def Compute_Transport_Plan(F, F_wgt, I, I_wgt , reg=1):
	'''
	Computes the optimal transport map, and transport cost, between two discrete distributions in R^2.
	The first two columns of each matrix represent locations of points, the third represents a weight.
	Weights should be normalized so that both distributions have the same total mass
	(in other words the last column of I and F should have the same sum).

	INPUTS: I - an Mx2 matrix I, where M is the number of points in the first distribution,
			F - Nx2 matrix, where N is the number of points for the second distribution.

	Output: (fun,Mout) - a tuple,
			fun - the transport cost
			Mout - an MxN matrix P that represents the transport plan.
			Each entry (i,j) represents the amount of mass transfered from point i in M to point j in N.
			This, if coded properly, should be a very sparse matrix. A transport cost C is also
			outputed, which gives the total transport cost.
	'''
	DistMat = metrics.pairwise.euclidean_distances( I, F )
	Mout = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg)


	temp = np.log(Mout)
	mask = np.isreal(np.log(Mout))
	temp[mask] = 0
	cost = np.sum(DistMat*Mout) + 1.0/reg*np.sum( temp*Mout )

	return cost, Mout


def transportGradient(Mout,I,I_wgt,F,F_wgt):
	'''
	computes gradient of transport problem in F
	'''
	output = np.zeros(F.shape)
	for i in range(len(I_wgt)):
		for j in range(len(F_wgt)):
			test = F[j,:] - I[i,:]

			d = np.linalg.norm(test)
			if d > .001:
				output[j,:] += Mout[i,j]*test/d
	return output


def optimizeF(I,I_wgt,Tinit,Finit,F_wgt):
	output = Finit.copy()

	newtonSteps = 20
	for i in range(newtonSteps):
		output -= transportGradient(Tinit,I,I_wgt,output,F_wgt)

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



def gradientDescentOptimalTransport(Iin,I_wgt,Fin,F_wgt,precinct_data=None, T1 = None, videoFlag=False, reg=10):
	'''
	'''
	I = Iin
	F = Fin
	Nini = len(I_wgt)
	uinit = np.ones(Nini)/Nini

	#Initial optimization step, mostly to initialize u
	DistMat = metrics.pairwise.euclidean_distances( I, F )
	Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg,uinit)

	Mout_list = []
	F_list = []

	#For the moment I'm hard coding the number of Newton steps
	newtonSteps = 20
	lineSearchN = 5
	lineSearchRange = 2
	costVec = np.zeros(lineSearchN)
	stepVector = np.linspace(0,lineSearchRange,lineSearchN)

	if videoFlag:
		#TODO: 	Use the interpolation function to go between Tinitial and Mout
		#		Then plot it.

		# set up initial figure, enable interactive plotting
		sb.set_palette("cool", n_colors=n_districts)
		plt.ion()
		fig, ax = sb.plt.subplots(1,1, figsize=(7,5))
		col = plot_polygon_collection(ax, precinct_data.geometry.values, values=precinct_data.current_district.values)
		fig.savefig('../maps/Colorado/movie/GD_alg_'+str(0)+'.png')
		plt.pause(1e-5)


		#This block of code does the interpolation between Tinit and Mout
		#Interpolation occurs by finding the precincts with lowest distance to correct precincts
		TCurrent = Tinit.copy()
		Tout = np.array([np.sum(Mout[i,:])*(Mout[i,:] == np.max(Mout[i,:])) for i in range(Nini)])

		D = metrics.pairwise.euclidean_distances( I, I )

		ind = 0
		NFrame = 50


		#This for loop and the arglist after essentially orders precincts by their distance
		#from precincts with the same correct final label. This was originally inside
		# the while loop  (meaning that distance was from precincts with the correct
		#current label), but it was too slow that way.
		DTemp = np.zeros(D.shape)
		TMask = TCurrent*Tout
		for i in range(len(D)):
			#j = np.nonzero(Tout[i,:])[0][0]
			j = np.argmax(Tout[i,:])
			if np.sum(TMask[i,:]) != 0:
				DTemp[i,:] = 0
			else:
				DTemp[i,:] = D[i,:]*TMask[:,j]
		argList = np.argsort(np.min(DTemp + 1*(DTemp==0),axis=1))

		startVal = 0
		endVal = min(NFrame,Nini)

		#This while loop does the intepolation and plots it
		while np.max(np.abs(TCurrent-Tout))> .00000001:
			#print ind

			TCurrent[argList[startVal:endVal]] = Tout[argList[startVal:endVal]]

			startVal += NFrame
			endVal = min(startVal+NFrame, Nini)

			precinct_data['district_initialSteps'+str(ind)] = np.array( [np.argmax(i) for i in TCurrent])
			filename = '../maps/Colorado/movie/Initial_Steps' + str(ind) + '.png'

			# update colors on map
			col.set_array(precinct_data['district_initialSteps'+str(ind)])


			plt.pause(1e-5)
			fig.savefig(filename)

			ind += 1






	for i_step in range(1,newtonSteps):
		#Compute an optimal plan, given a certain I,F
		DistMat = metrics.pairwise.euclidean_distances( I, F )
		Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg,u)
		precinct_data['district_iter'+str(i_step)] = np.array( [np.argmax(i) for i in Mout])

		if videoFlag:

			filename = '../maps/Colorado/movie/GD_alg_' + str(i_step) + '.png'

			# update colors on map
			col.set_array(precinct_data['district_iter'+str(i_step)])

			# for i_F in range(n_districts):
			# 	ax.scatter(F[i_F,0], F[i_F,1], color='black', marker='*', s=120, alpha=.2)

			plt.pause(1e-5)
			fig.savefig(filename)


		#Compute the Gradient in F
		Grad = transportGradient(Mout,I,I_wgt,F,F_wgt)

		Mlist = []
		for j in range(lineSearchN):
			#Exectue line search in the direction Grad
			DistMat = metrics.pairwise.euclidean_distances( I, F - stepVector[j]*Grad )
			Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg,u)
			costVec[j] = cost
			Mlist.append(Mout)


		ind = np.argmin(costVec)
		print np.min(costVec)
		F = F - stepVector[ind]*Grad

		F_list.append(F)
		Mout_list.append(Mlist[ind])
	# if videoFlag:
	# 	file_names = ['tempfig' + str(i) + '.png' for i in range(newtonSteps)]
	# 	images = [Image.open(fn) for fn in file_names]
	# 	writeGif('OutputMovie.GIF', images, duration=0.2)


	DistMat = metrics.pairwise.euclidean_distances( I, F )
	Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg,uinit)

	Mout_list.append(Mout)
	return Mout_list,cost,F_list




def make_scatterfig(I, F_opt, classify_vec, filename=None):
	'''
	Makes a simple scatter plot of points in I and F, colored so that points in I mapping to
	a point in F have the same color.
	INPUTS: I - precinct centers, plotted as dots
	  		F - distict offices, plotted as stars
			classify_vec - vector giving the district of each point in I, numpy array
	'''

	# numerical parameter
	N = len(F_opt)

	# Build figure and set colors
	fig, ax = sb.plt.subplots(1,1, figsize=(10,5))
	sb.set_style("whitegrid")
	palette = sb.color_palette("hls", N)
	for i in range(N):
		mask = classify_vec == i
		ax.scatter(I[mask,0], I[mask,1], color=palette[i], marker='o', alpha=.2)

	for i in range(N):
		ax.scatter(F_opt[i,0], F_opt[i,1], color=palette[i]	, marker='*', s=120)
		ax.scatter(F_opt[i,0], F_opt[i,1], color='black', marker='*', s=120, alpha=.2)


	# set plot limits
	plt.xlim([min(np.min(F[:,0]), np.min(I[:,0]))-.5, max(np.max(F[:,0]), np.max(I[:,0]))+.5 ])
	plt.ylim([min(np.min(F[:,1]), np.min(I[:,1]))-.5, max(np.max(F[:,1]), np.max(I[:,1]))+.5 ])

	if filename!= None:
		fig.savefig(filename, bbox_inches='tight', dpi=1200)
		plt.close()
	else:
		plt.show()


def make_map(geo_df, F_opt, filename=None):
	'''
	This function makes a map from GeoDataFrame, colored by voter district.
	INPUTS: geo_df   - dataframe from state shapefile, GeoPandas GeoDataFrame
			F_opt    - location of district offices, numpy array
			filename - if provided, saves file at path given by filename, string
	'''
	# set up figure and color palette
	fig, ax = sb.plt.subplots(1,2, figsize=(15,5))

	# make the map
	CD_plot = geo_df.plot(column='current_district'ax=ax[0], cmap='gist_rainbow',linewidth=0)
	precinct_plot = geo_df.plot(column='district_iter19',ax=ax[1], cmap='gist_rainbow',linewidth=0)

	ax[0].set_title('Current Congressional Districts')
	ax[1].set_title('Optimal Transport Districts')

	# plot stars (transparent gray so color is a darker shade of surrounding color)
	for i in range(len(F_opt)):
		ax[1].scatter(F_opt[i,0], F_opt[i,1], color='black', marker='*', s=40, alpha=.3)

	# save figure
	if filename!= None:
		fig.savefig(filename, bbox_inches='tight', dpi=800)
		plt.close()
	else:
		plt.show()


# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# SCRIPT BEGINS HERE
# ------------------------------------------------------------------------------
# ------------------------------------------------------------------------------
# read in data, set some parameters
# ------------------------------------------------------------------------------
precinct_data = geo.GeoDataFrame.from_file('../data/colorado/CO_final.shp')
state         = precinct_data.GEOID10.values[0][0:2]


# reading in census data
census_geo  = geo.GeoDataFrame.from_file('../data/census/tl_2015_us_cd114/tl_2015_us_cd114.shp')
census_geo['state'] = [i[0:2] for i in census_geo.GEOID.values]
census_geo = census_geo[census_geo.state==state]

n_districts    = len( census_geo )
n_precincts   = len( precinct_data )


# ------------------------------------------------------------------------------
# assign precincts to the congressional district with largest intersection
# ------------------------------------------------------------------------------
dist_geos = census_geo.geometry.values
prec_geos = precinct_data.geometry.values

I     = precinct_data[['INTPTLON_1','INTPTLAT_1']].values
I_wgt = precinct_data.POP100.values/precinct_data.POP100.values.sum()

#precinct_data['current_district'] = [np.argmax(np.array([dist_geos[i].intersection(j).area for i in range(n_districts)])) for j in prec_geos]



#This code was run and now is pickled below
Tinit = np.array([[I_wgt[j]*dist_geos[i].contains(shapely.geometry.Point(I[j,:])) for j in range(n_precincts)] for i in range(n_districts)])
Tinit = Tinit.T
#pickle.dump( [Tinit], open( "../data/colorado/Tinit.p", "wb" ) )

[Tinit] = pickle.load( open( "../data/colorado/Tinit.p", "rb" ) )


precinct_data['current_district'] = np.argmax(Tinit,axis=1)

#[1*np.array([dist_geos[i].contains(shapely.geometry.Point(I[j,:]) for i in range(n_districts)])) for j in range(n_precincts)]
#precinct_data['current_district'] = np.argmax(Tinit)
#precinct_data['current_district'] = [np.argmax(1*np.array([dist_geos[i].contains(shapely.geometry.Point(I[j,:])) for i in range(n_districts)])) for j in range(n_precincts)]


# ------------------------------------------------------------------------------
# precinct location and weights (I), district office location and weights (F)
# ------------------------------------------------------------------------------
#I     = precinct_data[['INTPTLON_1','INTPTLAT_1']].values



Finit = I[ np.random.randint( 0,n_precincts,n_districts ) ]

F_wgt = np.ones((len(Finit),))/len(Finit)

F = optimizeF(I,I_wgt,Tinit,Finit,F_wgt)

#F     = I[ np.random.randint( 0,n_precincts,n_districts ) ]



# ------------------------------------------------------------------------------
# find optimal district offices and corresponding districts
# ------------------------------------------------------------------------------

OptimalPlan_steps, cost, OptimalOffices_steps = gradientDescentOptimalTransport(I,I_wgt,F,F_wgt, precinct_data, T1 = Tinit,videoFlag=True)
output = OptimalPlan_steps[-1]
F_opt  = OptimalOffices_steps[-1]




# ------------------------------------------------------------------------------
# using optimal F, determine which points belongs to each office
# or load previous result
# ------------------------------------------------------------------------------
if 1:

	pickle.dump( [OptimalPlan_steps,I,OptimalOffices_steps,precinct_data], open( "../data/colorado/precinct_data.p", "wb" ) )

else:
	[OptimalPlan_steps, I,F_opt,precinct_data] = pickle.load( open( "../data/colorado/precinct_data.p", "rb" ) )
	output = OptimalPlan_steps[-1]
	F_opt = OptimalOffices
	districts = np.array( [np.argmax(i) for i in output])


	tic = time.time()
	plt.ion()
	fig, ax = sb.plt.subplots(1,1, figsize=(7,5))

	col = plot_polygon_collection(ax, precinct_data.geometry.values, values=precinct_data.district_iter0.values)
	for i in range(20):
		F_opt = OptimalOffices_steps[i]
		col.set_array(precinct_data['district_iter'+str(i)])

		for i_F in range(len(F_opt)):
			ax.scatter(F_opt[i_F,0], F_opt[i_F,1], color='black', marker='*', s=120, alpha=.2)

		plt.pause(.2)
		# fig.savefig('test'+str(i)+'.pdf')
		print i, time.time() - tic




# ------------------------------------------------------------------------------
# make scatter plot and map
# ------------------------------------------------------------------------------
make_scatterfig(I, F_opt, precinct_data.district.values, filename='figures/CO_dots.pdf')
make_map(census_geo, precinct_data, F_opt, filename='figures/CO_map_Test.pdf')
