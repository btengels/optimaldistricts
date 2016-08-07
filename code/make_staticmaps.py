from __future__ import division

import numpy as np

import pandas as pd
import geopandas as geo
import pickle
import shapely
from shapely.ops import cascaded_union
from shapely.ops import unary_union

import time
import transport_plan_functions as tpf
import sklearn.metrics as metrics

from matplotlib import pyplot as plt
import seaborn as sns

import os


def make_folder(path):
	'''
	I am too lazy to make 50 folders + subfolders, so...
	'''
	try: 
	    os.makedirs(path)
	except OSError:
	    if not os.path.isdir(path):
	        raise


def draw_boundary(geo_df, district_group, n_districts, ax, outside=False):
	'''
	Draws outside boundary for collection of precincts
	'''
	# inside boundaries
	linewidth=.15
	color='black'
	alpha = .4
	district_list = []
	for i in range(n_districts):		
		df = geo_df[geo_df[district_group]==i]
		district = cascaded_union( df.geometry.values)		
		district_list.append(district.buffer(1e-7))

		# plot light circle outside each district
		if type(district)==shapely.geometry.polygon.Polygon:
			x,y = district.exterior.xy
			ax.plot(x,y, linewidth=linewidth, color=color, alpha=alpha)			
		else:
			for d in district:
				x,y = d.exterior.xy
				ax.plot(x,y, linewidth=linewidth, color=color, alpha=alpha)	

	# outside boundaries
	district = cascaded_union( district_list )
	if type(district)==shapely.geometry.polygon.Polygon:
		x,y = district.exterior.xy
		ax.plot(x,y, linewidth=linewidth, color=color, alpha=alpha)			
	else:
		for d in district:
			x,y = d.exterior.xy
			ax.plot(x,y, linewidth=linewidth, color=color, alpha=alpha)	

	return district


def plot_state(geo_df, district_iter, filename, F_opt=None):
	'''
	'''
	n_districts = int( geo_df.CD_2010.max() )+1
	lon_range = geo_df.INTPTLON10.max() - geo_df.INTPTLON10.min()
	lat_range = geo_df.INTPTLAT10.max() - geo_df.INTPTLAT10.min()

	# make figure/axis objects. Plot current districts
	fig, ax = sns.plt.subplots(1,1, figsize=(lon_range,lat_range*1.3))
	# plot outside line of each district
	state = draw_boundary(geo_df[geo_df.POP_TOTAL>0], district_iter, n_districts, ax)	
	CD_plot = geo_df[geo_df.POP_TOTAL>0].plot(column=district_iter, ax=ax, cmap='Set1',linewidth=0, alpha=.8)	
	
	df = geo_df[(geo_df.POP_TOTAL==0)]
	lakes = df.plot(ax=ax, color='blue',linewidth=0.1, alpha=.4)	

	# get dimensions to ensure grids are consistent between old/new plots	
	minx, miny, maxx, maxy = state.bounds
	ax.set_xlim(minx,maxx)
	ax.set_ylim(miny,maxy)

	# turn off axis ticklabels (longitude and latitude numbers aren't interesting)
	ax.get_xaxis().set_ticklabels([])
	ax.get_yaxis().set_ticklabels([])	

	# plot stars (transparent gray so color is a darker shade of surrounding color)
	if F_opt is not None:
		for i in range(len(F_opt)):
			ax.scatter(F_opt[i,0], F_opt[i,1], color='black', marker='*', s=30, alpha=1)	
			# ax.annotate(str(i), (F_opt[i,0], F_opt[i,1]))	
	
	if filename_stub is not None:
		fig.savefig(filename, bbox_inches='tight', dpi=100)
		plt.close()


def get_optimal_districts(state, videoFlag=False, reg=10, black_param=0, random_start=False ):
	'''
	'''
	# read in census data
	precinct_data = pd.read_pickle( '../Data-Files/'+state+'/precinct_data.p')

	# weight each precinct by its population (so each cluster has approx the same population)
	precinct_location = precinct_data[['INTPTLON10','INTPTLAT10','BLACK_PCT']].values
	precinct_wgt      = precinct_data.POP_TOTAL.values/precinct_data.POP_TOTAL.values.sum()

	# keep the number of districts the same as in data
	n_districts    = int( precinct_data.CD_2010.max() )+1
	n_precincts   = len( precinct_data )

	print 'load stuff'
	if random_start == True:
		# randomly select initial districts, all districts have equal weight
		Office_location0 = precinct_location[ np.random.randint( 0,n_precincts,n_districts ) ]
		
	else:
		# use centroid of current district as initial guess
		Office_location0 = np.zeros((n_districts,3))
		for i in range(n_districts):		
			df = precinct_data[precinct_data['CD_2010']==i]
			district = cascaded_union( df.geometry.values)
			Office_location0[i,0] = district.representative_point().coords.xy[0][0]
			Office_location0[i,1] = district.representative_point().coords.xy[1][0]
			Office_location0[i,2] = df.POP_BLACK.sum()/df.POP_TOTAL.sum()

	Office_wgt = np.ones((n_districts,))/n_districts

	# initial transport plan: i.e. the measure of population assigned to each district 
	office_starts = np.zeros((n_precincts,n_districts)) 
	for i_d,dist in enumerate( precinct_data.CD_2010.values):
		office_starts[ i_d,int(dist) ] = precinct_wgt[i_d]

	print 'office location'
	# distance of each precinct to its center (aka, "office")
	DistMat_travel       = metrics.pairwise.euclidean_distances( precinct_location[:,0:2], Office_location0[:,0:2] )
	
	# demographic distance
	DistMat_demographics = metrics.pairwise.euclidean_distances( np.atleast_2d( precinct_location[:,2]).T, np.atleast_2d( Office_location0[:,2] ).T)

	# total distance
	DistMat = DistMat_travel + black_param*DistMat_demographics

	# compute current cost function (to measure before/after improvements)
	Office_location = tpf.optimizeF( precinct_location, precinct_wgt, office_starts, Office_location0, Office_wgt, DistMat )
	temp = np.log(office_starts)	
	temp[np.isreal(np.log(office_starts))] = 0
	reg = 10
	cost0 = np.sum(DistMat*office_starts) + 1.0/reg*np.sum( temp*office_starts )

	# compute optimal districts, its associated cost and gradient descent path
	OptimalPlan_steps, cost, OptimalOffices_steps = tpf.gradientDescentOptimalTransport(precinct_location, precinct_wgt, Office_location, Office_wgt, precinct_data,Tinit=office_starts, videoFlag=videoFlag, reg=reg, black_param=black_param)	
	F_opt  = OptimalOffices_steps[-1]

	return cost0, cost, precinct_data, Office_location0, F_opt



states = {
			'AL':'Alabama',
			# 'AK':'Alaska',
			'AZ':'Arizona',
			'AR':'Arkansas',
			# 'CA':'California',
			'CO':'Colorado',
			'CT':'Connecticut',
			'FL':'Florida',
			'GA':'Georgia',
			'HI':'Hawaii',
			'ID':'Idaho',
			'IL':'Illinois',
			'IN':'Indiana',
			'IA':'Iowa',
			'KS':'Kansas',
			'KY':'Kentucky',
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
			'NH':'New Hampshire',
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
			'WY':'Wyoming'
			}

if __name__ == '__main__':
	
	for state in states:
		for black_param in [0,.25,.5,.75,1,1.5,2.5]:

			print state						
			make_folder('../maps/'+state)			
			make_folder('../maps/'+state+'/static')
			make_folder('../tables/'+state)
			
			cost0, cost, precinct_data_result, F0, F_opt = get_optimal_districts(state, black_param=black_param)
			filename_stub = '../maps/'+state+'/static/'+str(black_param).replace('.','_')+'_'			
			if black_param == 0:
				plot_state(precinct_data_result, 'CD_2010', filename_stub+'before.png')
			plot_state(precinct_data_result, 'district_iter19', filename_stub+'after.png', F_opt=F_opt)			
			
			precinct_data_result['cost0'] = cost0
			precinct_data_result['cost_final'] = cost
			precinct_data_result.to_pickle('../tables/'+state+'/results_'+str(black_param)+'.p')


		# final_tables = pd.DataFrame(state_df).T	
		# final_tables.to_pickle('../tables/before_after_stats_blackparam'+str(black_param)+'.p')

