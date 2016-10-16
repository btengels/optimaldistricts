
import numpy as np
import pandas as pd
import geopandas as geo
import pickle

from matplotlib import pyplot as plt
import seaborn as sns

from bokeh.io import figure, output_file, show, output_notebook
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




states = {	#'AL':'01',
			# 'AZ':'04',
			# 'CA':'06',
			# 'CO':'08', 
			'CT':'09', 
			# 'FL':'12', 
			# 'GA':'13', 
			# # 'HI':'15', 
			# 'IA':'19', 
			# # 'ID':'16', 
			# 'IL':'17', 
			# 'IN':'18', 
			# 'KS':'20', 
			# 'LA':'22', 
			# 'MA':'25', 
			# 'MD':'24', 
			# 'MI':'26', 
			# 'MN':'27', 
			# 'MO':'29', 
			# 'MS':'28', 
			# 'NC':'37', 
			# 'NE':'31', 
			# # 'NH':'33', 
			# 'NJ':'34', 
			# 'NM':'35', 
			# 'NV':'32', 
			# 'NY':'36',
			# 'OH':'39', 
			# 'OK':'40', 
			# # 'OR':'41', #problem here
			# 'PA':'42', 
			# # 'RI':'44', #problem here
			# 'SC':'45', 
			# 'TN':'47', 
			# 'TX':'48', 
			# # 'UT':'49', #problem here
			# 'VA':'51',# used https://github.com/vapublicaccessproject/va-precinct-maps instead
			# 'WA':'53', 
			# 'WI':'55',
			}



if __name__ == '__main__':


	pcnct_df_new.to_pickle('../tables/' + state + '/results_' + str(alphaW) + '.p')


	TOOLS="resize,crosshair,pan,wheel_zoom,box_zoom,reset,tap,previewsave,box_select,poly_select,lasso_select"

	output_file("color_scatter.html", title="color_scatter.py example")


	p = figure(tools=TOOLS)
	p.scatter(x, y, radius=radii, fill_color=colors, fill_alpha=0.6, line_color=None)

	output_file('test.html')
	save(p)

	show(p)  # open a browser	

	state_df={}
	for black_param in [0]:
		statelist = states.keys()
		for state in states:
		# while len(statelist)>0:
			# state = statelist[0]
			# try:
			print state
			make_folder('../maps/'+state)
			make_folder('../maps/'+state+'/dynamic')				

			filename = '../maps/'+state+'/static/foo.png'
			cost0, cost, precinct_data_result, F0, F_opt = get_optimal_districts( filename, state, black_param=black_param)
			make_singlemaps(precinct_data_result, filename_stub='../maps/'+state+'/dynamic/'+str(black_param).replace('.','_')+'_')				
					
			# except:
				# pass

