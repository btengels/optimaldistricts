from __future__ import division

import numpy as np

import pandas as pd
import geopandas as geo
import pickle
import shapely

import time
import transport_plan_functions as tpf
import sklearn.metrics as metrics

from matplotlib import pyplot as plt
import seaborn as sns

import os


def make_folder(path):
	'''
	I am too lazy to make 50 folders + subfolders, so 
	'''
	try: 
	    os.makedirs(path)
	except OSError:
	    if not os.path.isdir(path):
	        raise


def make_singlemaps(geo_df, F_opt, filename_stub=None):
	'''
	This function makes a map from GeoDataFrame, colored by voter district.
	INPUTS: geo_df   - dataframe from state shapefile, GeoPandas GeoDataFrame
			F_opt    - location of district offices, numpy array
			filename - if provided, saves file at path given by filename, string
	'''
	# set up figure and color palette
	lon_range = geo_df.INTPTLON10.max() - geo_df.INTPTLON10.min()
	lat_range = geo_df.INTPTLAT10.max() - geo_df.INTPTLAT10.min()


	fig, ax = sns.plt.subplots(1,1, figsize=(lon_range,lat_range*1.3))
	CD_plot = geo_df.plot(column='current_district',ax=ax, cmap='gist_rainbow',linewidth=0)
	# ax.set_title('Current Congressional Districts')
	xlim = ax.get_xlim()
	ylim = ax.get_ylim()
	
	if filename_stub != None:
		fig.savefig(filename_stub+'before.pdf', bbox_inches='tight', dpi=300)
		fig.savefig(filename_stub+'before.png', bbox_inches='tight', dpi=100)



	fig, ax = sns.plt.subplots(1,1, figsize=(lon_range,lat_range*1.3))
	CD_plot = geo_df.plot(column='district_iter19',ax=ax, cmap='gist_rainbow',linewidth=0)
	# ax.set_title('Optimal Transport Districts')
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)
	# plot stars (transparent gray so color is a darker shade of surrounding color)
	for i in range(len(F_opt)):
		ax.scatter(F_opt[i,0], F_opt[i,1], color='black', marker='*', s=40, alpha=.3)	

	if filename_stub != None:
		fig.savefig(filename_stub+'after.pdf', bbox_inches='tight', dpi=300)			
		fig.savefig(filename_stub+'after.png', bbox_inches='tight', dpi=100)			

	# save figure
	# if filename!= None:
	# 	fig.savefig(filename, bbox_inches='tight', dpi=800)
	# 	plt.close()
	# else:
	# 	plt.show()



def make_jointmap(geo_df, F_opt, filename=None):
	'''
	This function makes a map from GeoDataFrame, colored by voter district.
	INPUTS: geo_df   - dataframe from state shapefile, GeoPandas GeoDataFrame
			F_opt    - location of district offices, numpy array
			filename - if provided, saves file at path given by filename, string
	'''
	# set up figure and color palette
	fig, ax = sns.plt.subplots(1,2, figsize=(15,5),sharex=True, sharey=True)

	# make the map
	CD_plot = geo_df.plot(column='current_district',ax=ax[0], cmap='gist_rainbow',linewidth=0)
	precinct_plot = geo_df.plot(column='district_iter19',ax=ax[1], cmap='gist_rainbow',linewidth=0)

	# ax[0].set_title('Current Congressional Districts')
	# ax[1].set_title('Optimal Transport Districts')

	# plot stars (transparent gray so color is a darker shade of surrounding color)
	for i in range(len(F_opt)):
		ax[1].scatter(F_opt[i,0], F_opt[i,1], color='black', marker='*', s=40, alpha=.3)

	# save figure
	if filename!= None:
		fig.savefig(filename, bbox_inches='tight', dpi=800)
		plt.close()
	else:
		plt.show()



def get_optimal_districts( filename, state, makeMap=True, videoFlag=False, reg=10, black_param=0 ):
	'''
	'''
	
	precinct_data = pd.read_pickle( '../Data-Files-simple/'+state+'/precinct_data_demographics.p')
	precinct_data = precinct_data[precinct_data.total_pop.values>0]

	precinct_data['Black_percent'] = precinct_data.Black/precinct_data.total_pop

	precinct_location = precinct_data[['INTPTLON10','INTPTLAT10','Black_percent']].values
	precinct_wgt      = precinct_data.total_pop.values/precinct_data.total_pop.values.sum()

	n_districts    = int( precinct_data.current_district.max() )+1
	n_precincts   = len( precinct_data )

	print 'load stuff'
	Office_location0 = precinct_location[ np.random.randint( 0,n_precincts,n_districts ) ]
	Office_wgt = np.ones((len(Office_location0),))/len(Office_location0)

	# measure of population assigned to each district (we assign all of the precinct to its current district)
	office_starts = np.zeros((n_precincts,n_districts)) # what is going on here?
	for i_d,dist in enumerate( precinct_data.current_district.values):
		office_starts[ i_d,int(dist) ] = precinct_wgt[i_d]

	print 'office location'
	DistMat_travel       = metrics.pairwise.euclidean_distances( precinct_location[:,0:2], Office_location0[:,0:2] )
	DistMat_demographics = metrics.pairwise.euclidean_distances( np.atleast_2d( precinct_location[:,2]).T, np.atleast_2d( Office_location0[:,2] ).T)
	DistMat = DistMat_travel + black_param*DistMat_demographics

	
	Office_location = tpf.optimizeF( precinct_location, precinct_wgt, office_starts, Office_location0, Office_wgt, DistMat )

	print 'here'
	temp = np.log(office_starts)	
	temp[np.isreal(np.log(office_starts))] = 0
	reg = 10
	cost0 = np.sum(DistMat*office_starts) + 1.0/reg*np.sum( temp*office_starts )


	OptimalPlan_steps, cost, OptimalOffices_steps = tpf.gradientDescentOptimalTransport(precinct_location, precinct_wgt, Office_location, Office_wgt, precinct_data,Tinit=office_starts, videoFlag=videoFlag, reg=reg, black_param=black_param)
	
	F_opt  = OptimalOffices_steps[-1]


	# tic = time.time()
	# plt.ion()
	# fig, ax = sns.plt.subplots(1,1, figsize=(7,5))
	# precinct_data['district_iter0'] = precinct_data.current_district

	# col = tpf.plot_polygon_collection(ax, precinct_data.geometry.values, values=precinct_data.district_iter0)
	# for i in range(len(OptimalOffices_steps)):
	# 	F_opt = OptimalOffices_steps[i]
	# 	col.set_array(precinct_data['district_iter'+str(i)])

	# 	for i_F in range(len(F_opt)):
	# 		ax.scatter(F_opt[i_F,0], F_opt[i_F,1], color='black', marker='*', s=120, alpha=.2)

	# 	plt.pause(.2)
	# 	# fig.savefig('test'+str(i)+'.pdf')
	# 	print i, time.time() - tic

	if makeMap==True:
		make_jointmap(precinct_data, F_opt, filename=filename)

	return cost0, cost, precinct_data, F_opt



states = {	'AK':'02', #problem here
			'AL':'01',
			'AZ':'04',
			'CA':'06',
			'CO':'08', 
			'CT':'09', 
			'DE':'10', 
			'FL':'12', 
			'GA':'13', 
			'HI':'15', 
			'IA':'19', 
			'ID':'16', 
			'IL':'17', 
			'IN':'18', 
			'KS':'20', 
			'LA':'22', 
			'MA':'25', 
			'MD':'24', 
			'MI':'26', 
			'MN':'27', 
			'MO':'29', 
			'MS':'28', 
			'NC':'37', 
			'ND':'38', 
			'NE':'31', 
			'NH':'33', 
			'NJ':'34', 
			'NM':'35', 
			'NV':'32', 
			'NY':'36',
			'OH':'39', 
			'OK':'40', 
			# 'OR':'41', #probrlem here
			'PA':'42', 
			# 'RI':'44', #problem here
			'SC':'45', 
			'SD':'46', 
			'TN':'47', 
			'TX':'48', 
			# 'UT':'49', #problem here
			'VA':'51',# used https://github.com/vapublicaccessproject/va-precinct-maps instead
			'VT':'50', 
			'WA':'53', 
			'WI':'55',
			'WY':'56'
			}


state_df={}
black_param=0
statelist = states.keys()
# for state in states:
while len(statelist)>0:
	state = statelist[0]
	try:
		print state
		make_folder('../maps/'+state)
		make_folder('../maps/'+state+'/movie_files')
		make_folder('../maps/'+state+'/static')

		filename = '../maps/'+state+'/static/before_after.png'
		cost0, cost, precinct_data_result, F_opt = get_optimal_districts( filename, state, videoFlag=False,makeMap=False,black_param=0)
		# make_singlemaps(precinct_data_result, F_opt, filename_stub='../maps/'+state+'/static/')	
		
		# # 'total_pop',\
		# # 'Hispanic_Latino',\
		# # 'White',\
		# # 'Black',\
		# # 'American_Indian',\
		# # 'Asian',\
		# # 'Hawaiian_Islander',\
		# # 'Other',\

		old_df = precinct_data_result.groupby('current_district').agg(np.sum)
		new_df = precinct_data_result.groupby('district_iter19').agg(np.sum)
		
		#$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
		# this is where you decide what stats will appear in the tables...
		state_df[state] = {	'cost_improve': (cost0-cost)/cost0,
							'old_minblack':np.min(old_df.Black.values/old_df.total_pop.values),
							'old_medblack':np.median(old_df.Black.values/old_df.total_pop.values),
							'old_meanblack':np.median(old_df.Black.values/old_df.total_pop.values),
							'old_maxblack':np.max(old_df.Black.values/old_df.total_pop.values),
							'new_minblack':np.min(new_df.Black.values/new_df.total_pop.values),
							'new_medblack':np.median(new_df.Black.values/new_df.total_pop.values),
							'new_meanblack':np.median(new_df.Black.values/new_df.total_pop.values),
							'new_maxblack':np.max(new_df.Black.values/new_df.total_pop.values),						
							 }
	
		statelist.pop()
	except:
		pass



final_tables = pd.DataFrame(state_df).T	
final_tables.to_pickle('../tables/before_after_stats_blackparam'+str(black_param)+'.p')
	# old_df = precinct_data_result.groupby('current_district').agg(np.sum)
	# new_df = precinct_data_result.groupby('district_iter19').agg(np.sum)
	# state_results={}
	# state_results[state] = {'old_black_min': np.min(old_df.Black.values/old_df.total_pop.values),
	# 						'new_black_min': np.min(new_df.Black.values/new_df.total_pop.values),
	# 						'old_black_max': np.max(old_df.Black.values/old_df.total_pop.values),
	# 						'old_black_max': np.max(old_df.Black.values/old_df.total_pop.values),
	# 						}
	# T = pd.DataFrame(np.array([old_df.Black.values/old_df.total_pop.values, new_df.Black.values/new_df.total_pop.values]).T)
	# T.columns = ['Black_old','Black_new']
	# print T
