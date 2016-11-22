from __future__ import division
import scipy as sp
import numpy as np

from sklearn.metrics.pairwise import euclidean_distances
import matplotlib.pyplot as plt
import pandas as pd

import geopandas as geo
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import shapely
import pickle
import time
import os


def optimizeF(I, I_wgt, Finit, F_wgt, transp_map, DistMat):
	'''
	This function computes the optimal office location given the current office 
	location, transport map, distance matrix, and precinct locations.

	INPUTS:
	----------------------------------------------------------------------------
	I: numpy array, Nx3 array of precinct coordinates
	I_wgt: numpy array, Nx1 array of precinct population weights
	Finit: numpy array, Mx3 array of office coordinates
	F_wgt: numpy array, Mx1 array of office population weights
	transp_map: numpy array, NxM array for distribution of I_wgt over M offices
	DistMat: numpy array, NxM pairwise distrance matrix between I and F arrays

	OUTPUTS:
	----------------------------------------------------------------------------
	F_opt: numpy array, Mx3 array of office coordinates
	'''
	F_opt = Finit.copy()

	newtonSteps = 30
	for i in range(newtonSteps):
		F_opt -= _transportGradient(I, I_wgt, F_opt, F_wgt, transp_map, DistMat)	

		# keep offices inside the state's bounding box	
		latmin = I[:, 0].min()
		lonmin = I[:, 1].min()
		latmax = I[:, 0].max()
		lonmax = I[:, 1].max()		

		F_opt[:,0] = np.maximum(np.minimum(F_opt[:, 0], latmax), latmin)
		F_opt[:,1] = np.maximum(np.minimum(F_opt[:, 1], lonmax), lonmin)

		# keep demographic percentages between zero and one
		F_opt[:,2] = np.maximum(np.minimum(F_opt[:, 2], 1), 0)

	return F_opt


def distance_metric(I_in, F_in, alphaW):
	'''
	Computes pairwise 'distance' between precincts (I_in) and offices (F_in)
	based on both geographic and demographic features. An office is close to a 
	precinct along the demographic dimension if the district* demographic stats
	are similar to the precinct demographic stats. This keeps precincts with
	similar demographic features within the same district (to a degree). 

	INPUTS:
	----------------------------------------------------------------------------
	I_in: np.array, Nx3 array of precinct coordinates
	F_in: np.array, Mx3 array of office coordinates
	alphaW: float, scaling parameter for demographic distance

	OUTPUTS:
	----------------------------------------------------------------------------
	DistMat: numpy array, pairwise distance between precincts and offices
	'''
	# demographic distance
	I_demopgraphics = np.atleast_2d(I_in[:, 2]).T
	F_demographics = np.atleast_2d(F_in[:, 2]).T
	Dist_demographics = euclidean_distances(I_demopgraphics, F_demographics)

	# geographic distance
	Dist_travel = euclidean_distances(I_in[:, 0:2], F_in[:, 0:2])

	# total distance
	DistMat = Dist_travel + alphaW*Dist_demographics	

	return DistMat


def gradientDescentOT(Iin, I_wgt, Fin, F_wgt, reg=20, alphaW=0):
	'''
	This function is a simple gradient descent optimization routine which 
	minimizes the weighted sum of distances from precincts (I) and the 
	congressional office (F). The cost function also includes a regularization 
	penalty on the transport map itself (simple transport maps are good).

	INPUTS:
	----------------------------------------------------------------------------
	Iin: numpy array, Nx2 array giving lat/lon coordinates for precincts
	I_wgt: numpy array, Nx1 array giving population weights for precincts
	Fin: numpy array, Mx2 array giving lat/lon coordinates for offices
	F_wgt: numpy array, Mx1 array giving population weights for offices
	reg: float (default=20), regularization parameter
	alphaW: float (default=0), weight on demographics in distance metric

	OUTPUTS:
	----------------------------------------------------------------------------
	opt_district: numpy array, vector of districts for precincts in Iin
	F: numpy array, Mx2 array giving optimal lat/lon coordinates for offices
	cost: float, transport cost (sum of distances + regularization penalty term)
	'''
	n_districts = len(Fin)
	I = Iin
	F = Fin
	Nini = len(I_wgt)
	uinit = np.ones(Nini)/Nini

	#Initial optimization step, mostly to initialize u
	DistMat = distance_metric(I, F, alphaW)
	transp_map, u, cost = _computeSinkhorn(I_wgt, F_wgt, DistMat, reg, uinit)

	#Hard coding the number of Newton steps for now. 
	newtonSteps = 30
	lineSearchN = 20
	lineSearchRange = 1
	costVec = np.zeros(lineSearchN)
	stepsize_vec = np.linspace(0,lineSearchRange,lineSearchN)

	# gradient descent algorithm
	for i_step in range(1, newtonSteps):
		# Compute an optimal transport plan, given I and a particular F
		DistMat = distance_metric(I, F, alphaW)		
		transp_map, u, cost = _computeSinkhorn(I_wgt, F_wgt, DistMat, reg, u)

		# Compute the Gradient in F
		Grad = _transportGradient(I, I_wgt, F, F_wgt, transp_map, DistMat)

		# find optimal step size (will step in direction of gradient)
		for j in range(lineSearchN):
			DistMat = distance_metric(I, F - stepsize_vec[j]*Grad, alphaW)
			transp_map, u, cost = _computeSinkhorn(I_wgt, F_wgt, DistMat, reg, u)
			costVec[j] = cost 

		# find the optimal step size and adjust F accordingly
		ind = np.argmin(costVec)
		# print(np.min(costVec))
		F -= stepsize_vec[ind]*Grad

	DistMat = distance_metric(I, F, alphaW)
	transp_map, u, cost = _computeSinkhorn(I_wgt, F_wgt, DistMat, reg, uinit)	
	opt_district = np.array([np.argmax(i) for i in transp_map])

	return opt_district, F, cost


def make_adjacency_matrix(df):
	'''
	This function takes a geopandas DataFrame and computesan adjacency matrix 
	from the polygons in the 'geometry' column. For states with many precincts,
	this adjacency matrix can become quite large. Uses symmetry to speed up 
	computation.

	INPUTS:
	----------------------------------------------------------------------------
	df: GeoDataFrame, dataframe with column 'geometry' containing
	    shapely polygons for voter precincts.

	OUTPUTS:
	----------------------------------------------------------------------------
	adj_matix: 	numpy array, binary array. Element (i,j) = 1 if the i_th
				precint and j_th precint intersect or share a common 
				boundary. Otherwise (i,j) = 0. 
	'''	
	# build adjacency matrix
	adj_matrix = np.zeros((len(df), len(df)))
	for i, p in enumerate(df.geometry.values):		
		adj_matrix[i, i:] = df[i:].intersects(p)
		adj_matrix[i:, i] = adj_matrix[i, i:].flatten()

	return adj_matrix


def check_contiguity(adj_matrix, district_vec):
	'''
	This function takes an adjacency matrix and a vector denoting which
	row/column belongs to common districts, and determines if the 
	entire state is comprised of contiguous districts. 

	INPUTS:
	----------------------------------------------------------------------------
	adj_matrix: numpy array, element (i,j) = 1 if precincts i and j are adjacent
				(even at a point, as long as shapely can detect it)
	district_vec: numpy array, vector denoting the district of each row of the 
				adjacency matrix

	OUTPUTS:
	----------------------------------------------------------------------------
	contiguity: boolean, Is True if all precincts are contiguous, and is false 
				if any precinct is not contiguous
	'''
	contig_true = True
	i_d = 0
	n_districts = len(np.unique(district_vec))

	# check if each district is contiguous
	while contig_true and i_d < n_districts:
		
		# slice out relevant subset of adjacency matrix
		mask = district_vec == i_d
		dist_adj_mat = adj_matrix[mask, :]
		dist_adj_mat = dist_adj_mat[:, mask]

		n_prec = mask.sum()

		# If a district is contiguous, then you should be able to walk from 
		# any precinct to another in at least n_prec steps or less. 
		prec_vec = np.zeros((n_prec, 1))
		prec_vec[0] = 1
		
		for i in range(n_prec):
			prec_vec = np.dot(dist_adj_mat, prec_vec) > 0

		# are all precincts within the district connected?
		contig_true = prec_vec.sum() == n_prec
		i_d += 1

	return contig_true	

	
def _computeSinkhorn(I_wgt, F_wgt, Distmat, reg, uin):
	'''
	This function uses the Sinkhorn algorithm to update the transport map given
	the current population weights, Distance map, regularization parameter, and 

	INPUTS:
	----------------------------------------------------------------------------
	I_wgt: numpy array, Nx1 array of precinct population weights
	F_wgt: numpy array, Mx1 array of office population weights
	DistMat: numpy array, NxM pairwise distrance matrix between I and F arrays
	reg: float, regularization parameter
	uin: numpy array, vector used in projection step of Sinkhorn algorithm.

	OUTPUTS:
	----------------------------------------------------------------------------
	transp_map: numpy array, NxM array for distribution of I_wgt over M offices
	u: numpy array, vector resulting from projection step of Sinkhorn algorithm.
	   Not important by itself. Using as uin in future calls is a big speed up.
	cost: float, value of cost function evaluated at optimal transp_mat
	'''
	# init data
	Nini = len(I_wgt)
	Nfin = len(F_wgt)

	numItermax = 200

	# assume that no distances are null except the diagonal of Distmat
	u = uin.copy()
	uprev=np.zeros(Nini)

	# transport map
	K = np.exp(-reg*Distmat)	
	Kp = K*(1/np.atleast_2d(I_wgt).T)
	transp = K.copy()

	# project map onto constraints repeatedly to find optimal transport map
	count = 0
	err=1
	while (err > 1e-5 and count < numItermax):
		if np.logical_or(np.any(np.dot(K.T, u) == 0), np.isnan(np.sum(u))):
			# we have reached machine precision
			# come back to previous solution and quit loop
			print('Infinity')
			if count!=0:
				u = uprev.copy()
			break

		uprev = u.copy()
		v = np.divide(F_wgt, np.dot(K.T,u))
		u = 1./np.dot(Kp, v)
		
		# Periodically checking convergency criterion 
		if count%3 == 0:		
			transp = np.atleast_2d(u).T*K*v		
			err = np.linalg.norm((np.sum(transp, axis=0) - F_wgt))**2
			# print(err)
		count += 1

	# after convergence is reached, back out transport map
	transport_map = np.atleast_2d(u).T*K*v
	temp = np.log(transport_map)
	mask = np.isreal(np.log(transport_map))
	temp[mask] = 0

	cost = np.sum(Distmat*transport_map) + 1.0/reg*np.sum(temp*transport_map)
	return transport_map, u, cost


def _transportGradient(I, I_wgt, F, F_wgt, trasport_map, DistMat):
	'''
	This function computes the gradient of transport problem in F. The gradient 
	of F is a matrix showing how the transport cost changes as coordinates in F 
	are peturbed.

	INPUTS:
	----------------------------------------------------------------------------
	trasport_map: numpy array, NxM array showing the population mass
		 		  allocated to each office in F
	I: numpy array, Nx3 array showing lat/lon/dem coordinates for precincts
	I_wgt: numpy array, Nx1 array showing population weight for precincts
	F: numpy array, Mx3 array showing lat/lon/dem coordinates for offices
	F_wgt: numpy array, Nx1 array showing population weight for offices
	Distmat: numpy array NxM array showing the pairwise distance from the nth 
			 precinct to the mth office

	OUTPUTS:
	----------------------------------------------------------------------------
	grad_F: numpy array, gradient of cost function with respect to F
	'''
	grad_F = np.zeros(F.shape)	
	
	for j in range(len(F_wgt)):

		# perturb office location
		distance = F[j, :] - I

		# trim out basicaly unchanged portions of transport map
		d = DistMat[:, j]
		mask = d > .001			

		# compute gradient
		map_new = np.tile(trasport_map[mask, j], (F.shape[1], 1)).T
		d_adjust = np.tile(d[mask], (F.shape[1], 1)).T		
		grad_F[j, :] = np.sum(map_new*distance[mask, :]/d_adjust, axis=0)
		
	return grad_F


def unpack_multipolygons(geo_df, impute_vals=True):
    '''
    Takes a vector of polygons and/or multipolygons. Returns an array of only 
    polygons by taking multipolygons and putting the underlying polygons in a 
    new vector. The "districts" vector contains info on the district for each 
    multipolygon/polygon. 

    INPUTS: 
    ----------------------------------------------------------------------------
    geo: numpy array, array of shapely polygons/multipolygons
    impute_vals: boolean(default=True) use area shares to impute populations or
    			 other variabels for polygons within a multipolygon. If false,
    			 population/political variables are duplicated across all 
    			 polygons within a multipolygon. 
    
    OUTPUTS:
    ----------------------------------------------------------------------------
    geo: numpy array, array of shapely polygons only, (no multipolygons)
    dists: numpy array, vector contianing info on each polygon
    '''
    repeat = False
    new_df = geo.GeoDataFrame()
    cols = [c for c in geo_df.columns if 'POP' in c or 'VAP' in c or 'PRE' in c]
    geo_df[cols] = geo_df[cols].astype(float)
    scale = 1 #

    # check geometry column for non-polygon objects (multipolygons)
    for i, row in geo_df.iterrows():
        row = row.T
        if type(row.geometry) == shapely.geometry.polygon.Polygon:
            new_df = new_df.append(row)

        else:
        	# unpack multipolygons, perhaps impute numerical variables
            multi_row = row.copy()
            total_area = multi_row.geometry.area            
            for poly in multi_row.geometry:

                if impute_vals is True:
                    scale = (poly.area/total_area)

                row.geometry = poly
                row[cols] *= scale
                row.ALAND10 *= scale
                row.area = poly.area        
                new_df = new_df.append(row)

    poly_type = shapely.geometry.polygon.Polygon
    poly_true = np.array([type(g) == poly_type for g in new_df.geometry.values])
    if poly_true.any() is False:
        new_df = unpack_multipolygons(new_df)

    return new_df	


def get_coords(T, xcoord=True):
	'''
	Takes a single shapely polygon and returns a list of its x or y coordinates.

	INPUTS
	----------------------------------------------------------------------------
	T: shapely Polygon or MultiPolygon, voter precinct or district
	xcoord: boolean (default=True), if true returns x coordinates, if false 
			returns y coordinates

	OUTPUT
	----------------------------------------------------------------------------	
	patchx: list, sequence of x coordinates corresponding to patchy
	patchy: list, sequence of y coordinates corresponding to patchx
	'''
	if type(T) == shapely.geometry.polygon.Polygon:
		patchx, patchy = T.exterior.coords.xy

	elif type(T) == shapely.geometry.multipolygon.MultiPolygon:
		T = T[0]
		patchx, patchy = T.exterior.coords.xy
	
	# can choose to return either the x or y coordinates 
	# (this is admittedly a bit odd but is required by pandas' "apply" function)
	if xcoord is True:
		return list(patchx)
	else:
		return list(patchy)


def make_folder(path):
    '''
    This function makes folders on the fly, checking first to see if the folder 
    is already there. 

    INPUTS
    ----------------------------------------------------------------------------    
    path: string, path where directory will be made

    OUTPUTS
    ----------------------------------------------------------------------------    
    None
    '''
    try: 
        # check to see if path already exists otherwise make folder
        os.makedirs(path)

    except OSError:
        if not os.path.isdir(path):
            raise

    return None



