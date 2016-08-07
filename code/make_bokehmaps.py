from __future__ import division

import numpy as np
import pandas as pd
import geopandas as geo
import pickle


import time
import transport_plan_functions as tpf
import sklearn.metrics as metrics

from matplotlib import pyplot as plt
import seaborn as sns

from bokeh.io import output_file, show, output_notebook
from bokeh.models import GeoJSONDataSource, ColumnDataSource
from bokeh.plotting import *

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

def get_xcoords(T):
    if type(T) == shapely.geometry.polygon.Polygon:
        patchx, patchy = T.exterior.coords.xy        

    elif type(T) == shapely.geometry.multipolygon.MultiPolygon:
        T = T[0]
        patchx, patchy = T.exterior.coords.xy
    return list(patchx)


def get_ycoords(T):
    if type(T) == shapely.geometry.polygon.Polygon:
        patchx, patchy = T.exterior.coords.xy        

    elif type(T) == shapely.geometry.multipolygon.MultiPolygon:
        T = T[0]
        patchx, patchy = T.exterior.coords.xy
    return list(patchy)




def make_singlemaps(geo_df, filename_stub=None):
	'''
	This function makes a map from GeoDataFrame, colored by voter district.
	INPUTS: geo_df   - dataframe from state shapefile, GeoPandas GeoDataFrame
			F_opt    - location of district offices, numpy array
			filename - if provided, saves file at path given by filename, string
	'''
	ndists = int(np.max(geo_df.current_district.values)+1)
	palette = np.array(sns.color_palette("Set1", n_colors=ndists, desat=.5).as_hex())
	geo_df['patchx'] = geo_df.geometry.apply(lambda row: get_xcoords(row))
	geo_df['patchy'] = geo_df.geometry.apply(lambda row: get_ycoords(row))
	geo_df['color1'] = palette[geo_df.current_district.values.astype(int)]
	geo_df['color2'] = palette[geo_df.district_iter19.values.astype(int)]

	source = ColumnDataSource(data=dict(
	    x = geo_df['patchx'].values.astype(list),
	    y = geo_df['patchy'].values.astype(list),
	    color1 = geo_df['color1'].values.astype(list),
	    color2 = geo_df['color2'].values.astype(list),
	    # district_name = geo_df['current_district'].values.astype(list),
	    # precinct_name = geo_df['NAME10'].values.astype(list),
	    # district_pop  = geo_df['district_pop'].values.astype(list),
	    # precinct_pop  = geo_df['POP100'].values.astype(list),
	))

	# tools for bokeh users
	TOOLS="pan,box_zoom,reset,save"

	# compute height/width of states to make maps are about the right shape
	lon_range = geo_df.INTPTLON10.max() - geo_df.INTPTLON10.min()
	lat_range = geo_df.INTPTLAT10.max() - geo_df.INTPTLAT10.min()
	
	p = figure(plot_width=int(lon_range*100), plot_height=int(lat_range*140), tools=TOOLS, toolbar_location='right')
	p.logo = None
	p.patches('x','y', source=source, 
	          fill_color='color1', fill_alpha=0.7, 
	          line_color=None, line_width=0.05,
	          line_alpha=.2)

	# Turn off tick labels
	p.axis.major_label_text_font_size = '0pt'  
	
	# Turn off tick marks 	
	p.axis.major_tick_line_color = None  # turn off major ticks
	p.axis[0].ticker.num_minor_ticks = 0  # turn off minor ticks
	p.axis[1].ticker.num_minor_ticks = 0

	# save output as html file
	output_file(filename_stub+'before.html')
	save(p)

	# change colors to final districts
	p.patches('x','y', source=source, 
          fill_color='color2', fill_alpha=0.7, 
          line_color=None, line_width=0.05,
          line_alpha=.2)

	# save final html file
	output_file(filename_stub+'after.html')
	save(p)


def get_optimal_districts( filename, state, videoFlag=False, reg=10, black_param=0 ):
	'''
	'''
	# read in census data
	precinct_data = pd.read_pickle( '../Data-Files-simple/'+state+'/precinct_data_demographics.p')
	precinct_data = precinct_data[precinct_data.total_pop.values>0]

	precinct_data['Black_percent'] = precinct_data.Black/precinct_data.total_pop

	# weight each precinct by its population (so each cluster has approx the same population)
	precinct_location = precinct_data[['INTPTLON10','INTPTLAT10','Black_percent']].values
	precinct_wgt      = precinct_data.total_pop.values/precinct_data.total_pop.values.sum()

	# keep the number of districts the same as in data
	n_districts    = int( precinct_data.current_district.max() )+1
	n_precincts   = len( precinct_data )

	print 'load stuff'
	# randomly select initial districts, all districts have equal weight
	Office_location0 = precinct_location[ np.random.randint( 0,n_precincts,n_districts ) ]
	Office_wgt = np.ones((len(Office_location0),))/len(Office_location0)

	# initial transport plan: i.e. the measure of population assigned to each district 
	office_starts = np.zeros((n_precincts,n_districts)) 
	for i_d,dist in enumerate( precinct_data.current_district.values):
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



states = {	'AL':'01',
			'AZ':'04',
			'CA':'06',
			'CO':'08', 
			'CT':'09', 
			'FL':'12', 
			'GA':'13', 
			# 'HI':'15', 
			'IA':'19', 
			# 'ID':'16', 
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
			'NE':'31', 
			# 'NH':'33', 
			'NJ':'34', 
			'NM':'35', 
			'NV':'32', 
			'NY':'36',
			'OH':'39', 
			'OK':'40', 
			# 'OR':'41', #problem here
			'PA':'42', 
			# 'RI':'44', #problem here
			'SC':'45', 
			'TN':'47', 
			'TX':'48', 
			# 'UT':'49', #problem here
			'VA':'51',# used https://github.com/vapublicaccessproject/va-precinct-maps instead
			'WA':'53', 
			'WI':'55',
			}



if __name__ == '__main__':

	state_df={}
	for black_param in [0,.05,.1,.15,.2,.25,.3,.5,.7,.9,1,1.5,2.5]:
		statelist = states.keys()
		for state in states:
		# while len(statelist)>0:
			# state = statelist[0]
			try:
				print state
				make_folder('../maps/'+state)
				make_folder('../maps/'+state+'/movie_files')
				make_folder('../maps/'+state+'/static')

				filename = '../maps/'+state+'/static/foo.png'
				cost0, cost, precinct_data_result, F0, F_opt = get_optimal_districts( filename, state, black_param=black_param)
				make_singlemaps(precinct_data_result, filename_stub='../maps/'+state+'/static/'+str(black_param).replace('.','_')+'_')				
					
			except:
				pass

