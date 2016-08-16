from __future__ import division

import numpy as np

import pandas as pd
import geopandas as geo
import pickle
import shapely
from shapely.ops import cascaded_union
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

import time
import transport_plan_functions as tpf
import sklearn.metrics as metrics

from matplotlib import pyplot as plt
import seaborn as sns
import os


def make_folder(path):
	'''
	I am too lazy to make 50 folders + subfolders. This makes all my 
	folders on the fly. 
	'''
	try: 
		# check to see if path already exists otherwise make folder
	    os.makedirs(path)
	except OSError:
	    if not os.path.isdir(path):
	        raise


def plot_patches(geoms, colors, ax, lw=.1):
	'''
	Plots precincts colored according to their congressional district.
	Uses matplotlib's PathCollection rather than geopandas' native plot() function.

	INPUTS: 
	-------------------------------------------------------------------------------
	geos: np.array, vector of polygons
	colors: np.array, vector indicating the color of each polygon	
	ax: matplotlib axis object
	lw: scalar, line width

	OUTPUTS: 
	-------------------------------------------------------------------------------	
	patches: matplotlib PathCollection object, we return this so we can plot the 
			 map once and then only worry about updating the colors later. 
	'''

	# make list of polygons (make sure they only have x and y coordinates)
	patches = []
	for poly in geoms:		
		a = np.asarray(poly.exterior)
		if poly.has_z:
			poly = shapely.geometry.Polygon(zip(*poly.exterior.xy))
		patches.append(Polygon(a))

	# make PatchCollection object
	patches = PatchCollection(patches)

	# set colors and linewidth
	patches.set_lw(lw)
	patches.set_color(colors)

	# plot on ax
	ax.add_collection(patches, autolim=True)
	ax.autoscale_view()
	return patches


def draw_boundary(geos, values, n_districts, ax):
	'''
	Draws outside boundary for collection of precincts. Currently, this is 
	slows down our code a lot and the results don't look super amazing...

	INPUTS: 
	-------------------------------------------------------------------------------
	geos: np.array, vector of polygons
	values: np.array, vector indicating the color of each polygon
	n_districts: scalar, number of congressional districts
	ax: matplotlib axis object

	OUTPUTS:
	-------------------------------------------------------------------------------
	ax - matplotlib axis object 
	'''
	# inside boundaries
	linewidth=.2
	color='black'
	alpha = .5

	district_list = []
	# interior boundaries
	for i in range(n_districts):	
		district = cascaded_union( geos[values==i] )
		district_list.append(district.buffer(1e-7))

	# outside boundary (takes union of interior boundaries)
	district = cascaded_union( district_list )
	district_list.append(district.buffer(1e-7))

	# draw the lines
	for dist in district_list:
		if type(dist)==shapely.geometry.polygon.Polygon:
			x,y = district.exterior.xy
			ax.plot(x,y, linewidth=linewidth, color=color, alpha=alpha)         
		else:
			for d in dist:
				x,y = d.exterior.xy
				ax.plot(x,y, linewidth=linewidth, color=color, alpha=alpha)

	return ax

def update_colors(geo_df, district_group, alpha=1):
	'''
	Takes a geopandas DataFrame and accomplishes the following:
	 1) unpacks multipolygons into a vector of simply polygons, called geoms
	 2) maintains corresponding vector of congressional districts for geoms
	 3) assigns a color to each congressional district
	 4) Figures out which precincts are mostly water, color = transparent blue

	INPUTS: 
	-------------------------------------------------------------------------------
	geo_df: geopandas DataFrame, contains precinct-level geometries and districts
	district_group: string, column indivating congressional districts
	alpha: scalar in [0,1], default alpha (transparency) for plots

	OUTPUTS:
	-------------------------------------------------------------------------------
	geoms: np.array, vector of polygons
	dists: np.array, indicates the congressional district for each polygon in geoms
	lakes: np.array, boolean indicator for whether district in geoms is inhabited
	colors: np.array, combines dists and lakes to provide a color for each precinct
	'''
	n_districts = int( geo_df.CD_2010.max() )+1
	
	# define the colormap
	cmap = plt.cm.Paired
	cmaplist = [cmap(i) for i in range(cmap.N)]		
	palette =[cmaplist[i] for i in range(0,len(cmaplist),int(len(cmaplist)/n_districts))]

	geoms, dists, lakes = tpf.unpack_multipolygons(geo_df.geometry.values, geo_df[district_group].values, geo_df.lakes.values)		
	colors = np.array( [ (palette[i][0], palette[i][1], palette[i][2], alpha) for i in dists] )
	colors[lakes==True] = (0,0,.9,.1)
	return geoms, dists, lakes, colors


def plot_state(geo_df, district_group, ax, fig, filename, F_opt=None):
	'''
	Function takes geopandas DataFrame and plots the precincts colored according
	to their congressional district. Saves figure at path "filename."

	INPUTS:
	-------------------------------------------------------------------------------
	geo_df: geopandas DataFrame
	district_group: string, column name in geo_df which contains each precinct's 
					congressional district
	ax: matplotlib axis object
	fig: matplotlib figure object, used to save final figure
	filename: string, path to saved figure
	F_opt: np.array, plots the location of district offices if provided
	'''
	
	# unpack "multipolygons" into indivual polygons
	geoms, dists, lakes, colors = update_colors(geo_df, district_group)
	patches = plot_patches(geoms,colors,ax,lw=.1)
	# ax = draw_boundary(geoms, dists, n_districts, ax)	

	# plot stars (transparent gray so color is a darker shade of surrounding color)
	if F_opt is not None:
		for i in range(len(F_opt)):
			ax.scatter(F_opt[i,0], F_opt[i,1], color='black', marker='*', s=30, alpha=1)	
			# ax.annotate(str(i), (F_opt[i,0], F_opt[i,1]))	

	if filename is not None:
		fig.savefig(filename, bbox_inches='tight', dpi=100)
		plt.close()			

	return patches


def get_optimal_districts(pcnct_df, alphaW, office_loc0, reg=10, random_start=False):
	'''
	Function takes a geopandas DataFrame and computes the congressional districts
	that minimize the total "distance" between precincts (rows in DataFrame) and 
	the "district office." 

	INPUTS: 
	-------------------------------------------------------------------------------
	pcnct_df: geopandas DataFrame listing precinct-level data for a given state
	alphaW: scalar, weight on non-geographic distance used in distance metric
	office_loc0: np.array, initial location of district offices
	reg: scalar, regularization term used in cost function
	random_start: boolean, if false algorithm uses office_loc0

	OUTPUTS:
	-------------------------------------------------------------------------------
	cost0: scalar, value of cost function at office_loc0
	cost: scalar, value of cost function at algorithm end
	pcnct_df: geopandas dataFrame, now with additional columns for each iteration 
	          of gradient descent algorithm
	F_opt: np.array, indicates the location of optimal office locations
	'''
	# weight each precinct by its population (so each cluster has approx the same population)
	pcnct_loc = pcnct_df[['INTPTLON10','INTPTLAT10','BLACK_PCT']].values
	pcnct_wgt = pcnct_df.POP_TOTAL.values/pcnct_df.POP_TOTAL.values.sum()

	# office weights (each office has the same share of population) 	
	office_wgt = np.ones((n_districts,))/n_districts

	# initial transport plan: i.e. the measure of population assigned to each district 
	office_starts = np.zeros((n_precincts,n_districts)) 
	for i_d,dist in enumerate( pcnct_df.CD_2010.values):
		office_starts[ i_d,int(dist) ] = pcnct_wgt[i_d]

	print 'office location'
	# distance of each precinct to its center (aka, "office")
	DistMat_travel       = metrics.pairwise.euclidean_distances( pcnct_loc[:,0:2], office_loc0[:,0:2] )
	
	# demographic distance
	DistMat_demographics = metrics.pairwise.euclidean_distances( np.atleast_2d( pcnct_loc[:,2]).T, np.atleast_2d( office_loc0[:,2] ).T)
	DistMat_demographics *= DistMat_travel.mean()

	# total distance
	DistMat = (1-alphaW)*DistMat_travel + alphaW*DistMat_demographics		

	# compute current cost function (to measure before/after improvements)
	office_loc = tpf.optimizeF( pcnct_loc, pcnct_wgt, office_starts, office_loc0, office_wgt, DistMat )
	temp = np.log(office_starts)	
	temp[np.isreal(np.log(office_starts))] = 0
	reg = 10
	cost0 = np.sum(DistMat*office_starts) + 1.0/reg*np.sum( temp*office_starts )

	# compute optimal districts, its associated cost and gradient descent path
	OptimalPlan_steps, cost, Offices_steps = tpf.gradientDescentOptimalTransport(pcnct_loc, pcnct_wgt, office_loc, office_wgt, pcnct_df, Tinit=office_starts, reg=reg, alphaW=alphaW)	
	F_opt  = Offices_steps[-1]

	return cost0, cost, pcnct_df, F_opt


states = {
			'AL':'Alabama',
			# 'AK':'Alaska',
			'AZ':'Arizona',
			'AR':'Arkansas',
			'CA':'California',
			# 'CO':'Colorado',
			'CT':'Connecticut',
			'FL':'Florida',
			'GA':'Georgia',
			'HI':'Hawaii',
			'ID':'Idaho',
			'IL':'Illinois',
			'IN':'Indiana',
			'IA':'Iowa',
			'KS':'Kansas',
			# 'KY':'Kentucky',
			'LA':'Louisiana',
			'MA':'Maine',
			'MD':'Maryland',
			'MA':'Massachusetts',
			'MI':'Michigan',
			'MN':'Minnesota',
			'MS':'Mississippi',
			'MO':'Missouri',
			# 'MT':'Montana',
			# 'NE':'Nebraska',
			'NV':'Nevada',
			# 'NH':'New Hampshire',
			'NJ':'New Jersey',
			'NM':'New Mexico',
			'NY':'New York',
			'NC':'North Carolina',
			# 'ND':'North Dakota',
			'OH':'Ohio',
			'OK':'Oklahoma',
			'OR':'Oregon',
			'PA':'Pennsylvania',
			# 'RI':'Rhode Island',
			'SC':'South Carolina',
			'SD':'South Dakota',
			'TN':'Tennessee',
			'TX':'Texas',
			'UT':'Utah',
			# 'VT':'Vermont',
			'VA':'Virginia',
			'WA':'Washington',
			'WV':'West Virginia',
			'WI':'Wisconsin',
			# 'WY':'Wyoming'
			}


if __name__ == '__main__':
	
	for state in states:
		print state

		# make map folders if not existent
		make_folder('../maps/'+state)
		make_folder('../maps/'+state+'/static')
		make_folder('../tables/'+state)

		# print initial map
		pcnct_df = pd.read_pickle( '../Data-Files/'+state+'/precinct_data.p')
		
		# keep the number of districts the same as in data
		n_districts    = int( pcnct_df.CD_2010.max() )+1
		n_precincts   = len( pcnct_df )				

		# initial guess for district offices
		random_start=False
		if random_start == True:
			# randomly select initial districts, all districts have equal weight
			pcnct_loc = pcnct_df[['INTPTLON10','INTPTLAT10','BLACK_PCT']].values			
			office_loc0 = pcnct_loc[ np.random.randint( 0,n_precincts,n_districts ) ]
			
		else:
			# use representative point of current district as initial guess
			office_loc0 = np.zeros((n_districts,3))
			for i in range(n_districts):				
				df = pcnct_df[pcnct_df['CD_2010'].values==i]			
				district = cascaded_union( df.geometry.values)
				office_loc0[i,0] = district.representative_point().coords.xy[0][0]
				office_loc0[i,1] = district.representative_point().coords.xy[1][0]
				office_loc0[i,2] = df.POP_BLACK.sum()/df.POP_TOTAL.sum()	

		# adjust scalaing for black_pct
		pcnct_df['BLACK_PCT'] /= 100

		# make figure/axis objects and plot	initial figure
		fig, ax = plt.subplots(1,1,subplot_kw=dict(aspect='equal'))		
		filename = '../maps/'+state+'/static/before.png'
		patches = plot_state(pcnct_df, 'CD_2010', ax, fig, filename)

		# get properties of initial plot to ensure updated maps are comparable
		xlim = ax.get_xlim()
		ylim = ax.get_ylim()

		# solve for optimal districts at different levels of alpha
		for alphaW in np.linspace(0,.5,51):
			print state, alphaW
			
			# get updated dataframe (includes districts for each iteration of CG algorithm)
			cost0, cost, pcnct_df_new, F_opt = get_optimal_districts(pcnct_df, alphaW, office_loc0)
								
			# update colors on existing figure for this solution of districting problem
			geoms, dists, lakes, colors = update_colors(pcnct_df_new, 'district_iter19')
			patches.set_color(colors)

			# make sure boundaries of figure are consistent across figures
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)

			# plot district offices and save figure
			stars =	ax.scatter(F_opt[:,0], F_opt[:,1], color='black', marker='*', s=30, alpha=.7)	
			filename = '../maps/'+state+'/static/'+str(alphaW).replace('.','_')+'_after.png'								
			fig.savefig(filename, bbox_inches='tight', dpi=100)
			stars.remove()

			# include before/after transport cost in resulting DataFrame
			pcnct_df_new['cost0'] = cost0
			pcnct_df_new['cost_final'] = cost
			pcnct_df_new.to_pickle('../tables/'+state+'/results_'+str(alphaW)+'.p')

		