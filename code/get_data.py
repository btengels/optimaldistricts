
from subprocess import call
import os
import geopandas as geo
from glob import glob
import numpy as np

def make_folder(path):
	'''
	I am too lazy to make 50 folders + subfolders, so...
	'''
	try: 
	    os.makedirs(path)
	except OSError:
	    if not os.path.isdir(path):
	        raise

def get_state_data(state,wget=False):
	prefix='../Data-Files/'+state
	make_folder(prefix)

	# import shape files
	if wget==True:
		call(['wget', '-P', prefix, 'ftp://autoredistrict.org/pub/shapefiles2/'+states[state]+'/2010/2012/vtd/tl*'])
	
	# read shape files into geopandas
	geo_path = glob(prefix+'/tl*.shp')[0]
	precinct_geo = geo.GeoDataFrame.from_file(geo_path)
	precinct_geo = precinct_geo[precinct_geo.CD_2000>=0]
	precinct_geo.CD_2000  = precinct_geo.CD_2000.astype(int)
	precinct_geo.CD_2010  = precinct_geo.CD_2010.astype(int)

	# add longitude and latitude
	lonlat =np.array([t.centroid.coords.xy for t in precinct_geo.geometry])
	precinct_geo['INTPTLON10'] = lonlat[:,0]
	precinct_geo['INTPTLAT10'] = lonlat[:,1]

	# make sure congressional districts are numbered starting at 0
	precinct_geo.CD_2000 -= precinct_geo.CD_2000.min()
	precinct_geo.CD_2010 -= precinct_geo.CD_2010.min()	

	# percent black in each precinct
	precinct_geo['BLACK_PCT'] = np.nanmin(precinct_geo['POP_BLACK']/precinct_geo['POP_TOTAL'],0)

	# simplify geometries for faster rendering
	precinct_geo.geometry = precinct_geo.geometry.simplify(.0001)	

	# pickle dataframe for future use
	precinct_geo.to_pickle( prefix+'/precinct_data.p' )	        

states = {
			'AL':'Alabama',
			'AK':'Alaska',
			'AZ':'Arizona',
			'AR':'Arkansas',
			'CA':'California',
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
			'MT':'Montana',
			'NE':'Nebraska',
			'NV':'Nevada',
			'NH':'New Hampshire',
			'NJ':'New Jersey',
			'NM':'New Mexico',
			'NY':'New York',
			'NC':'North Carolina',
			'ND':'North Dakota',
			'OH':'Ohio',
			'OK':'Oklahoma',
			'OR':'Oregon',
			'PA':'Pennsylvania',
			'RI':'Rhode Island',
			'SC':'South Carolina',
			'SD':'South Dakota',
			'TN':'Tennessee',
			'TX':'Texas',
			'UT':'Utah',
			'VT':'Vermont',
			'VA':'Virginia',
			'WA':'Washington',
			'WV':'West Virginia',
			'WI':'Wisconsin',
			'WY':'Wyoming'
			}


if __name__ == '__main__':
	
	# make main folder if it doesn't already exist
	make_folder('../Data-Files')

	# fill that folder with individual state folders containing precinct-level data
	for state in states:
		print state				
		get_state_data(state)
