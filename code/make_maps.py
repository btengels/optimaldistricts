from __future__ import division

import numpy as np

import pandas as pd
import geopandas as geo
import pickle
import shapely
from shapely.ops import cascaded_union
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from bokeh.io import output_file, show
from bokeh.models import GeoJSONDataSource, ColumnDataSource
from bokeh.plotting import *

import time
import transport_plan_functions as tpf
import sklearn.metrics as metrics

from matplotlib import pyplot as plt
import seaborn as sns
import os


def make_folder(path):
	'''
	I am too lazy to make 50 folders + subfolders. This makes all my folders on 
	the fly, checking first to see if the folder is already there. 

	INPUTS
	----------------------------------------------------------------------------	
	path: string, path where directory will be made
	'''
	try: 
		# check to see if path already exists otherwise make folder
	    os.makedirs(path)
	except OSError:
	    if not os.path.isdir(path):
	        raise


def get_xcoords(T):
	'''
	Takes a polygon and returns a list of its x  coordinates.

	INPUTS
	----------------------------------------------------------------------------
	T: shapely Polygon or MultiPolygon, voter precinct or district

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
	return list(patchx)


def get_ycoords(T):
	'''
	Takes a polygon and returns a list of its y coordinates.

	INPUTS
	----------------------------------------------------------------------------
	T: shapely Polygon or MultiPolygon, voter precinct or district

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
	return list(patchy)


def plot_patches(geoms, colors, ax, lw=.1):
	'''
	Plots precincts colored according to their congressional district.
	Uses matplotlib's PathCollection rather than geopandas' native plot() 
	function.

	INPUTS: 
	----------------------------------------------------------------------------
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
		district = cascaded_union(geos[values==i])
		district_list.append(district.buffer(1e-7))

	# outside boundary (takes union of interior boundaries)
	district = cascaded_union(district_list)
	district_list.append(district.buffer(1e-7))

	# draw the lines
	for dist in district_list:
		if type(dist)==shapely.geometry.polygon.Polygon:
			x, y = district.exterior.xy
			ax.plot(x, y, linewidth=linewidth, color=color, alpha=alpha)         
		else:
			for d in dist:
				x, y = d.exterior.xy
				ax.plot(x, y, linewidth=linewidth, color=color, alpha=alpha)
	return ax


def make_palette(n_districts, cmap=plt.cm.Paired):
	'''
	Takes matplotlib cmap object and generates a palette of n equidistant points
	over the cmap spectrum, returned as a list. 
	
	INPUTS:
	----------------------------------------------------------------------------
	n_districts: int, number of districts
	cmap: matplotlib colormap object, e.g. cmap = plt.cm.Paired
	'''
	
	# define the colormap
	cmaplist = [cmap(i) for i in range(cmap.N)]		
	palette =[cmaplist[i] for i in range(0, cmap.N, int(cmap.N/n_districts))]	
	return palette


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
	colors: np.array, combines dists to provide a color for each precinct
	'''
	n_districts = int(geo_df.CD_2010.max()) + 1
	
	palette = make_palette(n_districts)
	geoms, dists = tpf.unpack_multipolygons(geo_df.geometry.values, geo_df[district_group].values)		
	colors = np.array([(palette[i][0], palette[i][1], palette[i][2], alpha) for i in dists])
	return geoms, dists, colors


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
	geoms, dists, colors = update_colors(geo_df, district_group)
	patches = plot_patches(geoms, colors, ax, lw=.1)
	# ax = draw_boundary(geoms, dists, n_districts, ax)	

	
	# plot stars (transparent gray so color is a darker shade of surrounding color)
	if F_opt is not None:
		for i in range(len(F_opt)):
			ax.scatter(F_opt[i,0], F_opt[i,1], color='black', marker='*', s=30, alpha=1)	
			# ax.annotate(str(i), (F_opt[i,0], F_opt[i,1]))	

	if filename is not None:
		ax.set_yticklabels([])
		ax.set_xticklabels([])
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
	pcnct_loc = pcnct_df[['INTPTLON10', 'INTPTLAT10', 'BLACK_PCT']].values
	pcnct_wgt = pcnct_df.POP_TOTAL.values/pcnct_df.POP_TOTAL.values.sum()

	# office weights (each office has the same share of population) 	
	office_wgt = np.ones((n_districts,))/n_districts

	# initial transport plan: i.e. the measure of population assigned to each district 
	office_starts = np.zeros((n_precincts, n_districts)) 
	for i_d,dist in enumerate(pcnct_df.CD_2010.values):
		office_starts[i_d, int(dist)] = pcnct_wgt[i_d]	
	
	# distance of each precinct to its center (aka, "office")
	DistMat = tpf.distance_metric(pcnct_loc, office_loc0, alphaW)

	# compute current cost function (to measure before/after improvements)
	office_loc = tpf.optimizeF(pcnct_loc, pcnct_wgt, office_starts, office_loc0, office_wgt, DistMat)
	temp = np.log(office_starts)	
	temp[np.isreal(np.log(office_starts))] = 0
	cost0 = np.sum(DistMat*office_starts) + 1.0/reg*np.sum(temp*office_starts)

	# compute optimal districts, its associated cost and gradient descent path
	steps, cost, Office_steps = tpf.gradientDescentOT(pcnct_loc, pcnct_wgt, office_loc, office_wgt, pcnct_df,
													  Tinit=office_starts, reg=reg, alphaW=alphaW)	

	# final location of optimal office
	F_opt  = Office_steps[-1]

	return cost0, cost, pcnct_df, F_opt


def make_bokehplots(pcnct_df, alphaW):
	'''

	'''		
	# palette = make_palette(n_districts)
	palette = np.array(sns.color_palette("Set1", n_colors=n_districts, desat=.5).as_hex())

	# unpack multipolygons, make new (longer) dataframe
	dist0 = pcnct_df.CD_2010.values
	dist1 = pcnct_df.district_iter19.values
	geos, col1 = tpf.unpack_multipolygons(pcnct_df.geometry.values, dist0)
	geos, col2 = tpf.unpack_multipolygons(pcnct_df.geometry.values, dist1)
	df = pd.DataFrame({'geometry': geos, 'CD_2010': col1, 'district_iter19': col2})

	# 
	df['patchx'] = df.geometry.apply(lambda row: get_xcoords(row))
	df['patchy'] = df.geometry.apply(lambda row: get_ycoords(row))

	df['color1'] = [palette[i] for i in df.CD_2010]
	df['color2'] = [palette[i] for i in df.district_iter19]

	source = ColumnDataSource(data=dict(
		x = df['patchx'].values.astype(list),
		y = df['patchy'].values.astype(list),
		color1 = df['color1'].values.astype(list),
		color2 = df['color2'].values.astype(list),
		# district_name = pcnct_df['current_district'].values.astype(list),
		# precinct_name = pcnct_df['NAME10'].values.astype(list),
		# district_pop  = pcnct_df['district_pop'].values.astype(list),
		# precinct_pop  = pcnct_df['POP100'].values.astype(list),
	))

	# tools for bokeh users
	TOOLS = "pan,box_zoom,reset,save"

	# compute height/width of states to make maps are about the right shape
	lon_range = pcnct_df.INTPTLON10.max() - pcnct_df.INTPTLON10.min()
	lat_range = pcnct_df.INTPTLAT10.max() - pcnct_df.INTPTLAT10.min()
	
	# make bokeh figure
	p = figure(plot_width=500, 
			   plot_height=500, 
			   tools=TOOLS, toolbar_location='right')

	# p = figure(plot_width=int(lon_range*100), 
	# 		   plot_height=int(lat_range*140), 
	# 		   tools=TOOLS, toolbar_location='right')

	# remove bokeh logo
	p.toolbar.logo = None
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
	filename = '../maps/' + state + '/dynamic/before.html'
	output_file(filename)
	show(p)
	save(p)


	# change colors to final districts
	p.patches('x','y', source=source, 
          fill_color='color2', fill_alpha=0.7, 
          line_color=None, line_width=0.05,
          line_alpha=.2)

	# save final html file
	filename = '../maps/' + state + '/dynamic/' + str(alphaW).replace('.', '_') + '_after.html'
	output_file(filename)
	save(p)


def make_bokehplot_single(pcnct_df, district_col, filename):
	'''

	'''		
	# palette = make_palette(n_districts)
	palette = np.array(sns.color_palette("Set1", n_colors=n_districts, desat=.5).as_hex())

	# unpack multipolygons, make new (longer) dataframe
	dists = pcnct_df[district_col].values
	geos, col1 = tpf.unpack_multipolygons(pcnct_df.geometry.values, dists)	
	df = pd.DataFrame({'geometry': geos, 'district': col1})

	# 
	df['patchx'] = df.geometry.apply(lambda row: get_xcoords(row))
	df['patchy'] = df.geometry.apply(lambda row: get_ycoords(row))
	df['color1'] = [palette[i] for i in df.district]

	source = ColumnDataSource(data=dict(
		x = df['patchx'].values.astype(list),
		y = df['patchy'].values.astype(list),
		color1 = df['color1'].values.astype(list),
		# color2 = df['color2'].values.astype(list),
		# district_name = pcnct_df['current_district'].values.astype(list),
		# precinct_name = pcnct_df['NAME10'].values.astype(list),
		# district_pop  = pcnct_df['district_pop'].values.astype(list),
		# precinct_pop  = pcnct_df['POP100'].values.astype(list),
	))

	# tools for bokeh users
	TOOLS = "pan,box_zoom,reset,save"

	# compute height/width of states to make maps are about the right shape
	# lon_range = pcnct_df.INTPTLON10.max() - pcnct_df.INTPTLON10.min()
	# lat_range = pcnct_df.INTPTLAT10.max() - pcnct_df.INTPTLAT10.min()
	
	# make bokeh figure
	p = figure(plot_width=500, 
			   plot_height=500, 
			   tools=TOOLS, toolbar_location='right')

	# p = figure(plot_width=int(lon_range*100), 
	# 		   plot_height=int(lat_range*140), 
	# 		   tools=TOOLS, toolbar_location='right')

	# remove bokeh logo
	p.toolbar.logo = None
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
	output_file(filename)
	# show(p)
	save(p)


states = {
			'AL':'Alabama',
			# 'AK':'Alaska',
			'AZ':'Arizona',
			'AR':'Arkansas',
			'CA':'California',
			'CO':'Colorado',
			'CT':'Connecticut',
			'FL':'Florida',
			'GA':'Georgia',
			# 'HI':'Hawaii', #problem here
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
			'NE':'Nebraska',
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

	state_list = list(states.keys())
	state_list.sort()
	for state in state_list:
		print(state)

		# ----------------------------------------------------------------------
		# make map folders if not existent
		# ----------------------------------------------------------------------
		make_folder('../maps/' + state)
		make_folder('../maps/' + state + '/static')
		make_folder('../maps/' + state + '/dynamic')
		make_folder('../tables/' + state)

		# print initial map
		pcnct_df = pd.read_pickle('../Data-Files/' + state + '/precinct_data.p')
		
		# keep the number of districts the same as in data
		n_districts = int(pcnct_df.CD_2010.max()) + 1
		n_precincts = len(pcnct_df)

		# ----------------------------------------------------------------------
		# initial guess for district offices
		# ----------------------------------------------------------------------
		random_start = False
		if random_start == True:
			# randomly select initial districts, all districts have equal weight
			pcnct_loc = pcnct_df[['INTPTLON10', 'INTPTLAT10', 'BLACK_PCT']].values
			office_loc0 = pcnct_loc[np.random.randint(0, n_precincts, n_districts)]
			
		else:
			# use representative point of current district as initial guess
			office_loc0 = np.zeros((n_districts, 3))
			for i in range(n_districts):				
				df = pcnct_df[pcnct_df['CD_2010'].values==i]			
				district = cascaded_union(df.geometry.values)
				office_loc0[i,0] = district.representative_point().coords.xy[0][0]
				office_loc0[i,1] = district.representative_point().coords.xy[1][0]
				office_loc0[i,2] = df['POP_BLACK'].sum()/df['POP_TOTAL'].sum()	

		# ----------------------------------------------------------------------
		# adjust scalaing for black_pct
		# ----------------------------------------------------------------------
		pcnct_df['BLACK_PCT'] /= 100

		# ----------------------------------------------------------------------
		# make figure/axis objects and plot	initial figure
		# ----------------------------------------------------------------------
		fig, ax = plt.subplots(1, 1, subplot_kw=dict(aspect='equal'))

		# make sure all plots have same bounding boxes
		xlim = (pcnct_df.geometry.bounds.minx.min(), pcnct_df.geometry.bounds.maxx.max())
		ylim = (pcnct_df.geometry.bounds.miny.min(), pcnct_df.geometry.bounds.maxy.max())				
		ax.set_xlim(xlim)
		ax.set_ylim(ylim)

		# save figure for current districts
		filename = '../maps/' + state + '/static/before.png'
		patches = plot_state(pcnct_df, 'CD_2010', ax, fig, filename)

		# make before bokeh plot
		geolist = [cascaded_union(pcnct_df[pcnct_df.CD_2010 == i].geometry.values) for i in range(n_districts)]
		df = pd.DataFrame({'geometry': geolist, 'CD_2010':np.arange(n_districts)})					
		filename = '../maps/' + state + '/dynamic/before.html'
		make_bokehplot_single(df, 'CD_2010', filename)		

		# ----------------------------------------------------------------------
		# solve for optimal districts at different levels of alpha
		# ----------------------------------------------------------------------
		for alphaW in np.linspace(0, 2, 9):
			print(state, alphaW)
			
			# get updated dataframe (includes districts for each iteration of CG algorithm)
			cost0, cost, pcnct_df_new, F_opt = get_optimal_districts(pcnct_df, alphaW, office_loc0)
								
			# update colors on existing figure for this solution of districting problem
			geoms, dists, colors = update_colors(pcnct_df_new, 'district_iter19')
			patches.set_color(colors)

			# make sure boundaries of figure are consistent across figures
			ax.set_xlim(xlim)
			ax.set_ylim(ylim)

			# plot district offices and save figure
			stars =	ax.scatter(F_opt[:, 0], F_opt[:, 1], color='black', marker='*', s=30, alpha=.7)	
			filename = '../maps/' + state + '/static/' + str(alphaW).replace('.', '_') + '_after.png'
			fig.savefig(filename, bbox_inches='tight', dpi=300)
			stars.remove()

			# make bokeh plots			
			geolist = [cascaded_union(pcnct_df_new[pcnct_df_new.district_iter19 == i].geometry.values) for i in range(n_districts)]
			df = pd.DataFrame({'geometry': geolist, 'district_iter19':np.arange(n_districts)})			
			filename = '../maps/' + state + '/static/' + str(alphaW).replace('.', '_') + '_after.png'
			make_bokehplot_single(df, 'district_iter19', filename)

			# include before/after transport cost in resulting DataFrame
			pcnct_df_new['cost0'] = cost0
			pcnct_df_new['cost_final'] = cost
			pcnct_df_new.to_pickle('../tables/' + state + '/results_' + str(alphaW) + '.p')

			