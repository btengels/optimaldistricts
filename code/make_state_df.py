from __future__ import division

import numpy as np

import pandas as pd
import geopandas as geo
import pickle
import shapely

import time


def make_state_census_dfs():
	'''
	Take the massive census dataframes of shape files and cut it up 
	into its individual states.
	'''	
	# congressional districts
	congress_geo  = geo.GeoDataFrame.from_file('../Data-Files/census/tl_2015_us_cd114/tl_2015_us_cd114.shp')	
	tract_geo_df  = geo.GeoDataFrame.from_file('../Data-Files/census/Tract_2010Census_DP1/Tract_2010Census_DP1.shp')		

	congress_geo.geometry = congress_geo.geometry.buffer(0)
	tract_geo_df.geometry = tract_geo_df.geometry.buffer(0)

	congress_geo['state'] = [i[0:2] for i in congress_geo.GEOID.values]
	tract_geo_df['state'] = [i[0:2] for i in tract_geo_df.GEOID10.values]

	for state in states:	
		print 'census', state

		state_id = states[state]				
		congress_geo_df = congress_geo[congress_geo.state==state_id]
		congress_geo_df.to_pickle('../Data-Files/'+state+'/congress_geo_'+state+'.p')

		tract_geo_df = tract_geo_df[tract_geo_df.state==state_id]
		tract_geo_df.to_pickle( '../Data-Files/'+state+'/tract_geo_'+state+'_df.p')



def get_current_districts(I_points,dist_geos):
	'''
	Function that gets current census district for each precinct
	'''
	# precinct_data['current_district'] = [np.argmax(np.array([dist_geos[i].intersection(j).area for i in range(n_districts)])) for j in prec_geos]
	# # Tinit = np.array([[I_wgt[j]*dist_geos[i].contains(shapely.geometry.Point(I[j,:])) for j in range(n_precincts)] for i in range(n_districts)])
	# Tinit = Tinit.T


	# tic = time.time()
	n_precincts = len( I_points )
	n_districts = len( dist_geos )
	current_district = np.zeros((n_precincts,))
	for i in range( n_precincts ):

		# first see if the congressional district intersects the centroid of the precinct
		in_district = False
		i_d = 0
		while not in_district and i_d < n_districts:

			in_district = dist_geos[i_d].contains( I_points[i] ) 
			# print i, i_d, in_district
			i_d+=1
	
		if i_d <= n_districts:
			current_district[i] = int( i_d-1 )

		else: 
			# in some cases (islands), the centroids are not the most useful. 
			# so find the congressional district w/ the greatest intersection. 
			intersect_vector = np.zeros(n_districts)			
			for i_d in range(n_districts):

				intersect_vector[i_d] = dist_geos[i_d].intersection( I_points[i] ).area
		
			current_district[i] = int( np.argmax( intersect_vector ) )

	return current_district

	

def make_precinct_df(state, save_pickled_df=True):
	'''
	'''
	# ------------------------------------------------------------------------------------------
	# read in precinct data
	# ------------------------------------------------------------------------------------------
	precinct_data = geo.GeoDataFrame.from_file('../Data-Files/'+state+'/'+state+'_final.shp')
	lonlat =np.array([t.centroid.coords.xy for t in precinct_data.geometry])
	precinct_data['INTPTLON10'] = lonlat[:,0]
	precinct_data['INTPTLAT10'] = lonlat[:,1]

	precinct_data.geometry = precinct_data.geometry.buffer(0)	
	state_id = states[state]
	

	# ------------------------------------------------------------------------------------------
	# reading in congressional district data
	# ------------------------------------------------------------------------------------------	
	congress_geo = pd.read_pickle( '../Data-Files/'+state+'/congress_geo_'+state+'.p')

	n_districts = len( congress_geo )
	n_precincts = len( precinct_data )


	# ------------------------------------------------------------------------------------------
	# reading in census tract demographic data
	# ------------------------------------------------------------------------------------------
	tract_geo_df = pd.read_pickle( '../Data-Files/'+state+'/tract_geo_'+state+'_df.p')


	# ------------------------------------------------------------------------------------------
	# get current districts
	# ------------------------------------------------------------------------------------------
	dist_geos = congress_geo.geometry.values
	I_points  = shapely.geometry.MultiPoint(precinct_data[['INTPTLON10','INTPTLAT10']].values.astype('float'))
	precinct_data['current_district'] = get_current_districts(I_points, dist_geos)	


	# ------------------------------------------------------------------------------------------
	# rename demographic variables of interest, toss out other census-tract variables
	# ------------------------------------------------------------------------------------------
	columns = {	'DP0040001': 'VA_total_pop' ,\
				'DP0040002': 'VA_male_pop' ,\
				'DP0040003': 'VA_female_pop' ,\
				'DP0110001': 'total_pop',\
				'DP0110002': 'Hispanic_Latino',\
				'DP0110011': 'White',\
				'DP0110012': 'Black',\
				'DP0110013': 'American_Indian',\
				'DP0110014': 'Asian',\
				'DP0110015': 'Hawaiian_Islander',\
				'DP0110016': 'Other',\
				'DP0110017': 'Two_or_more',\
				'DP0130001': 'Total_hh',\
				'DP0130002': 'hh_families',\
				'DP0130003': 'hh_families_families_w_children',\
				'DP0130004': 'hh_families_married',\
				'DP0130005': 'hh_families_married_w_children',\
				'DP0130006': 'hh_families_no_wife_present',\
				'DP0130007': 'hh_families_no_wife_present_w_children',\
				'DP0130008': 'hh_families_no_husband_present',\
				'DP0130009': 'hh_families_no_husband_present_w_children',\
				'DP0130010': 'hh_nonfamily'}

	tract_geo_df.rename(columns=columns, inplace=True)
	cols = [c for c in tract_geo_df.columns if c[0:2] != 'DP']
	tract_geo_df = tract_geo_df[cols]


	# ------------------------------------------------------------------------------------------
	# rename demographic variables of interest, toss out other census-tract variables
	# ------------------------------------------------------------------------------------------
	tract_array = tract_geo_df.geometry.values
	precinct_array = precinct_data.geometry.values


	tract_keys = [	'VA_total_pop' ,\
					'VA_male_pop' ,\
					'VA_female_pop' ,\
					'total_pop',\
					'Hispanic_Latino',\
					'White',\
					'Black',\
					'American_Indian',\
					'Asian',\
					'Hawaiian_Islander',\
					'Other',\
					'Two_or_more',\
					'Total_hh',\
					'hh_families',\
					'hh_families_families_w_children',\
					'hh_families_married',\
					'hh_families_married_w_children',\
					'hh_families_no_wife_present',\
					'hh_families_no_wife_present_w_children',\
					'hh_families_no_husband_present',\
					'hh_families_no_husband_present_w_children',\
					'hh_nonfamily']


	# ------------------------------------------------------------------------------------------
	# for each precinct, impute precinct-level demographics based on the weighted average of 
	# census-tracts. weights determined by fraction precinct intersecting with each census-tract
	# ------------------------------------------------------------------------------------------
	res_array = np.zeros( (n_precincts, len(tract_keys)) )
	mask = np.array([type(i)==shapely.geometry.polygon.Polygon for i in precinct_array])


	for i_p, precinct in enumerate(precinct_array):			 

		intersecting_tracts = np.where(np.array([precinct.intersects(t) for t in tract_array]) == True)[0]
		print i_p, i_p/n_precincts, intersecting_tracts
		frac_sum = 0
		for i_t in intersecting_tracts:			
			# find overlapping 
			frac_area = precinct.intersection(tract_array[i_t]).area/precinct.area
			
			for i_k, key in enumerate(tract_keys):
				
				tract_data = tract_geo_df[key].values[i_t]			
				res_array[i_p,i_k] += tract_data * frac_area


	for i_k, key in enumerate(tract_keys):
		precinct_data[key] = res_array[:,i_k]
	

	# ------------------------------------------------------------------------------------------
	# get current districts
	# ------------------------------------------------------------------------------------------
	# dist_geos = congress_geo.geometry.values
	# I_points  = shapely.geometry.MultiPoint(precinct_data[['INTPTLON10','INTPTLAT10']].values.astype('float'))
	# precinct_data['current_district'] = get_current_districts(I_points, dist_geos)
	
	if save_pickled_df:
		precinct_data.to_pickle('../Data-Files/'+state+'/'+state+'_precinct_data_demographics.p')
		
	return precinct_data



if __name__ == '__main__':
	
	tic = time.time()
	
	states = {	#'AK':'02', 
				# 'AL':'01',
				# 'AZ':'04',
				'CA':'06',
				'CO':'08', 
				# 'CT':'09', 
				'DE':'10', 
				# 'FL':'12', 
				# 'GA':'13', 
				# 'HI':'15', 
				# 'IA':'19', 
				# 'ID':'16', 
				# 'IL':'17', 
				# 'IN':'18', 
				'KS':'20', 
				# 'LA':'22', 
				# 'MA':'25', 
				# 'MD':'24', 
				'MI':'26', 
				# 'MN':'27', 
				'MO':'29', 
				'MS':'28', 
				'NC':'37', 
				# 'ND':'38', 
				'NE':'31', 
				# 'NH':'33', 
				# 'NJ':'34', 
				# 'NM':'35', 
				'NV':'32', 
				# 'NY':'36', 
				'OH':'39', 
				# 'OK':'40', 
				# 'OR':'41',
				'PA':'42', 
				# 'RI':'44', #problem here
				'SC':'45', 
				'SD':'46', 
				# 'TN':'47', 
				# 'TX':'48',#problem here 
				'UT':'49',
				# 'VA':'51',# used https://github.com/vapublicaccessproject/va-precinct-maps instead
				# 'VT':'50', 
				# 'WA':'53', 
				# 'WI':'55', #problem here - long/lat values are way off, but shapefiles have demographic data so perhaps we're ok
				# 'WY':'56'
				}
				


	# make_state_census_dfs()
	# state = 'CO'
	# make_precinct_df('CO')

	for state in states: 
		print state, (time.time()-tic)/60.0		
		make_precinct_df(state)



# <OPTION VALUE=01> 01 Alabama
# <OPTION VALUE=02> 02 Alaska
# <OPTION VALUE=04> 04 Arizona
# <OPTION VALUE=05> 05 Arkansas
# <OPTION VALUE=06> 06 California
# <OPTION VALUE=08> 08 Colorado
# <OPTION VALUE=09> 09 Connecticut
# <OPTION VALUE=10> 10 Delaware
# <OPTION VALUE=11> 11 District of Columbia
# <OPTION VALUE=12> 12 Florida
# <OPTION VALUE=13> 13 Georgia
# <OPTION VALUE=15> 15 Hawaii
# <OPTION VALUE=16> 16 Idaho
# <OPTION VALUE=17> 17 Illinois
# <OPTION VALUE=18> 18 Indiana
# <OPTION VALUE=19> 19 Iowa
# <OPTION VALUE=20> 20 Kansas
# <OPTION VALUE=21> 21 Kentucky
# <OPTION VALUE=22> 22 Louisiana
# <OPTION VALUE=23> 23 Maine
# <OPTION VALUE=24> 24 Maryland
# <OPTION VALUE=25> 25 Massachusetts
# <OPTION VALUE=26> 26 Michigan
# <OPTION VALUE=27> 27 Minnesota
# <OPTION VALUE=28> 28 Mississippi
# <OPTION VALUE=29> 29 Missouri
# <OPTION VALUE=30> 30 Montana
# <OPTION VALUE=31> 31 Nebraska
# <OPTION VALUE=32> 32 Nevada
# <OPTION VALUE=33> 33 New Hampshire
# <OPTION VALUE=34> 34 New Jersey
# <OPTION VALUE=35> 35 New Mexico
# <OPTION VALUE=36> 36 New York
# <OPTION VALUE=37> 37 North Carolina
# <OPTION VALUE=38> 38 North Dakota
# <OPTION VALUE=39> 39 Ohio
# <OPTION VALUE=40> 40 Oklahoma
# <OPTION VALUE=41> 41 Oregon
# <OPTION VALUE=42> 42 Pennsylvania
# <OPTION VALUE=44> 44 Rhode Island
# <OPTION VALUE=45> 45 South Carolina
# <OPTION VALUE=46> 46 South Dakota
# <OPTION VALUE=47> 47 Tennessee
# <OPTION VALUE=48> 48 Texas
# <OPTION VALUE=49> 49 Utah
# <OPTION VALUE=50> 50 Vermont
# <OPTION VALUE=51> 51 Virginia
# <OPTION VALUE=53> 53 Washington
# <OPTION VALUE=54> 54 West Virginia
# <OPTION VALUE=55> 55 Wisconsin
# <OPTION VALUE=56> 56 Wyoming

