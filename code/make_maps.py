from __future__ import division

import os
from subprocess import call
from glob import glob

import numpy as np
import pandas as pd
import geopandas as geo
import seaborn as sns
import shapely
import pickle

import matplotlib as mpl
from matplotlib import pyplot as plt

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

from bokeh.io import output_file
from bokeh.models import ColumnDataSource as CDS
from bokeh.plotting import *
from bokeh.models import HoverTool
from bokeh import mpl as bokeh_mpl

import transport_plan_functions as tpf


def get_state_data(state, wget=False):
    '''
    This function downloads the data from autoredistrict.org's ftp. After some 
    minor cleaning, the data is saved as a geopandas DataFrame.

    NOTE: currently the shape files on autoredistrict's ftp are districts
    instead of precincts as before. Don't use wget. 

    INPUT:
    ----------------------------------------------------------------------------
    state: string, postal ID for state and key to "states" dictionary
    wget: boolian (default=False), whether to download new shape files.

    OUTPUT:
    ----------------------------------------------------------------------------
    None, but DataFrame is pickled in ../Data-Files/<state> alongside the shape
    files.
    '''
    # make folder if it doesn't already exist
    prefix = '../Data-Files/' + state
    tpf.make_folder(prefix)

    # import shape files
    # url = 'ftp://autoredistrict.org/pub/shapefiles2/' + states[state] + '/2010/2012/vtd/tl*'
    # if wget is True:
    #     call(['wget', '-P', prefix,
    #           ])

    # read shape files into geopandas
    geo_path = glob(prefix + '/tl*.shp')[0]
    geo_df = geo.GeoDataFrame.from_file(geo_path)
    geo_df.CD_2010 = geo_df.CD_2010.astype(int)

    # simplify geometries for faster image rendering
    # bigger number gives a smaller file size
    geo_df.geometry = geo_df.geometry.simplify(.006).buffer(0.001)

    # drops totals and other non-precinct observations
    geo_df = geo_df[geo_df.CD_2010 >= 0]

    # add longitude and latitude
    lonlat = np.array([t.centroid.coords.xy for t in geo_df.geometry])
    geo_df['INTPTLON10'] = lonlat[:, 0]
    geo_df['INTPTLAT10'] = lonlat[:, 1]

    # -------------------------------------------------------------------------
    # ADJUST ASPECT RATIO HERE:
    # -------------------------------------------------------------------------

    #TODO merge other code to this.


	# -------------------------------------------------------------------------
	# -------------------------------------------------------------------------    

    # make sure congressional districts are numbered starting at 0
    geo_df.CD_2010 -= geo_df.CD_2010.min()

    # correct a few curiosities
    if state in ['KY']:
        geo_df.drop(['POP_BLACK', 'POP_WHITE', 'POP_ASIAN', 'POP_HAWAII',
                           'POP_HISPAN', 'POP_INDIAN', 'POP_MULTI', 'POP_OTHER',
                           'POP_TOTAL'], 
                          axis=1, inplace=True)

        geo_df.rename(index=str, columns={'VAP_BLACK': 'POP_BLACK',
                                                'VAP_WHITE': 'POP_WHITE',
                                                'VAP_ASIAN': 'POP_ASIAN',
                                                'VAP_HAWAII': 'POP_HAWAII',
                                                'VAP_HISPAN': 'POP_HISPAN',
                                                'VAP_INDIAN': 'POP_INDIAN',
                                                'VAP_MULTI': 'POP_MULTI',
                                                'VAP_OTHER': 'POP_OTHER',
                                                'VAP_TOT': 'POP_TOTAL'},
                            inplace=True)

    # percent black in each precinct, account for precincts with zero population
    geo_df['BLACK_PCT'] = np.maximum(geo_df['POP_BLACK']/geo_df['POP_TOTAL'], 0)
    geo_df.loc[np.isfinite(geo_df['POP_TOTAL']) == False, 'BLACK_PCT'] = 0
    geo_df['BLACK_PCT'].replace('NaN', 0, inplace=True)

    # exclude shapes that have no land (bodies of water)
    geo_df = geo_df[geo_df.ALAND10.isnull() == False]
    geo_df[['ALAND10', 'AWATER10']] = geo_df[['ALAND10', 'AWATER10']].astype(int)    

    # trim out water polygons from dataframe
    water_cut = 20
    if state in ['NC', 'PA', 'NJ', 'CT', 'OH', 'TX', 'FL']:
        water_cut = 8

    if state in ['CA', 'MA', 'MI', 'WA', 'MN']:
        water_cut = 4

    if state in ['IL', 'WI', 'NY', 'MD', 'LA', 'AK']:
        water_cut = 2
    
    if water_cut < 20:
        geo_df['VTDST10'] = geo_df['VTDST10'].astype(str)
        geo_df = geo_df[ geo_df['VTDST10'].str.contains('ZZ') == False]
        geo_df = geo_df[ geo_df['VTDST10'].str.contains('BAY') == False]
        geo_df = geo_df[ geo_df['VTDST10'].str.contains('OCEAN') == False]    
        geo_df = geo_df[np.abs(geo_df['AWATER10']/geo_df['ALAND10']) < water_cut]

    # unpack multipolygons
    geo_df = tpf.unpack_multipolygons(geo_df)

    # pickle dataframe for future use
    pickle.dump(geo_df, open(prefix + '/precinct_data.p', 'wb'), protocol=2) 

    return None


def get_optimal_districts(pcnct_df, random_start=True, reg=25):
	'''
	This function takes a geopandas DataFrame and computes the set of 
	congressional districts that minimize the total "distance" between precincts
	(rows in DataFrame) and the district "offices" (centroids).

	INPUTS: 
	----------------------------------------------------------------------------
	pcnct_df: geopandas DataFrame listing precinct-level data for a given state
	random_start: boolean(default=True) random initial coordinates for offices
	reg: scalar, regularization term used in cost function

	OUTPUTS:
	----------------------------------------------------------------------------
	cost0: scalar, value of cost function at F_loc0
	cost: scalar, value of cost function at algorithm end
	pcnct_df: geopandas dataFrame, now with additional column 'final_district'
	F_opt: np.array, indicates the location of optimal office locations
	'''

	# weight each precinct by its population share
	n_districts = int(pcnct_df.CD_2010.max()) + 1
	n_precincts = len(pcnct_df)
	pcnct_loc = pcnct_df[['INTPTLON10', 'INTPTLAT10', 'BLACK_PCT']].values
	pcnct_pop = np.maximum(pcnct_df.POP_TOTAL.values, 20)
	pcnct_wgt = pcnct_pop/pcnct_pop.sum()

	# initial guess for district offices
	office_loc_list = []

	if random_start is True:
		num_starts = 25

		# randomly select initial districts, all districts have equal weight
		for i in range(num_starts):
			office = pcnct_loc[np.random.randint(0, n_precincts, n_districts)]
			office_loc_list.append(office) 
		
	else:
		# use most populated precinct of current district as initial guess
		office_loc0 = np.zeros((n_districts, 3))
		for i in range(n_districts):

			df = pcnct_df[pcnct_df['CD_2010'].values==i]
			# df['pop_area'] = df.POP_TOTAL/(df.area*1000)
			pop_argmax = df['POP_TOTAL'].argmax()

			office_loc0[i, 0] = df['INTPTLON10'].loc[pop_argmax]
			office_loc0[i, 1] = df['INTPTLAT10'].loc[pop_argmax]
			office_loc0[i, 2] = df['POP_BLACK'].sum()/df['POP_TOTAL'].sum()	
		
		office_loc_list.append(office_loc0) 

	# office weights (each office has the same share of population) 	
	F_wgt = np.ones((n_districts,))/n_districts

	# initial transport plan: i.e. the measure of pop assigned to each district 
	transp_map = np.zeros((n_precincts, n_districts)) 
	for i_d, dist in enumerate(pcnct_df.CD_2010.values):
		transp_map[i_d, int(dist)] = pcnct_wgt[i_d]	

	# solve for optimal districts as alphaW increases
	alphaW = 0
	contiguity = True
	while contiguity is True and alphaW < 1:
		print(state, alphaW)
		# initialize cost_best variable
		cost_best = 20

		# find the best result over (perhaps several) starting points
		for F_loc0 in office_loc_list:

			# distance of each precinct to its center (aka, "office")
			DistMat = tpf.distance_metric(pcnct_loc, F_loc0, alphaW)

			# evaluate cost function given current districts
			office_loc = tpf.optimizeF(pcnct_loc, pcnct_wgt,
									   F_loc0, F_wgt, 
									   transp_map, DistMat
									   )

			temp = np.log(transp_map)	
			temp[np.isreal(np.log(transp_map))] = 0
			cost0 = np.sum(DistMat*transp_map) + 1.0/reg*np.sum(temp*transp_map)

			# compute optimal districts, offices, and transport cost
			opt_dist, F_opt, cost = tpf.gradientDescentOT(pcnct_loc, pcnct_wgt, 
														  office_loc, F_wgt,
														  reg=reg, alphaW=alphaW
														 )		
			# check contiguity
			# contiguity = tpf.check_contiguity(pcnct_df, opt_dist)
			contig = True

			# update if we are the current best district and contiguous
			if cost < cost_best and contig is True:
				print(state, alphaW, cost_best)
				opt_district_best = opt_dist.copy()
				F_opt_best = F_opt.copy()
				cost0_best = cost0
				cost_best = cost
				alphaW_best = alphaW	

		# update alphaW
		if cost_best < 20:
			alphaW += .1
			opt_district_best = opt_district_best.astype(int)

			# NOTE: working on check_contiguity(), so exit at alphaW=0 for now
			contiguity = False
		else:
			contiguity = False

	return opt_district_best, F_opt_best, cost0_best, cost_best, alphaW_best


def make_state_maps(state, random_start=True):
	'''
	This function takes a state postal code and fetches the corresponding 
	pickled dataframe with precinct level data and geometries. The funciton then
	solves for the optimal district, and makes both static and bokeh-style plots
	for the state based on its current and optimal congressional districts.

	INPUTS: 
	----------------------------------------------------------------------------
	state: string, key to "states" dictionary
	random_start: boolean (default=False), whether to use random coordinates for
				  office locations (and choose the best of several iterations)

	OUTPUTS: 
	----------------------------------------------------------------------------
	df_list: python list, contains geopandas DataFrames. One for each value of 
			 alphaW. This will eventually just hold 2 dataframes for the current 
			 and optimal set of districts
	'''
	# make map folders if not existent
	tpf.make_folder('../maps/' + state)
	tpf.make_folder('../maps/' + state + '/static')
	tpf.make_folder('../maps/' + state + '/dynamic')	

	# read in data
	pcnct_df = pd.read_pickle('../Data-Files/' + state + '/precinct_data.p')

	# make palette to be used in all plots (try to keep similar colors apart)
	n_districts = len(pcnct_df.CD_2010.unique())
	palette = make_palette(n_districts, hex=True)
	np.random.shuffle(palette)

	# make figure/axis objects and plot	initial figure
	fig, ax = plt.subplots(1, 1, subplot_kw=dict(aspect='equal'))

	# make sure all plots have same bounding boxes
	xlim = (pcnct_df.geometry.bounds.minx.min(), pcnct_df.geometry.bounds.maxx.max())
	ylim = (pcnct_df.geometry.bounds.miny.min(), pcnct_df.geometry.bounds.maxy.max())				
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	# list to save district-level DataFrames
	df_list = []

	# open/save figure for current districts
	filename = '../maps/' + state + '/static/before.png'
	current_dists = pcnct_df['CD_2010'].values.astype(int)
	colors = np.array([palette[i] for i in current_dists])
	patches = plot_state(pcnct_df, colors, ax, fig, filename)
	
	# save html figure for current districts
	filename = '../maps/' + state + '/dynamic/before.html'
	district_df = make_bokeh_map(pcnct_df, 'CD_2010', palette, filename)
	df_list.append(district_df)

	# check to see if contiguity is broken in initial districts (islands?, etc.)
	# contiguity = tpf.check_contiguity(pcnct_df, 'CD_2010')

	# get optimal districts
	opt_dist, F_opt, cost0, cost, alphaW = get_optimal_districts(pcnct_df)

	# update dataframe with districts for each precinct
	pcnct_df['district_final'] = opt_dist

	# update colors on existing figure for optimal districting solution
	colors = np.array([palette[i] for i in opt_dist])
	patches.set_color(colors)

	# make sure bounding box is consistent across figures
	ax.set_xlim(xlim)
	ax.set_ylim(ylim)

	# plot district offices and save figure
	stars =	ax.scatter(F_opt[:, 0], F_opt[:, 1],
						 color='black',
						 marker='*', 
						 s=30, 
						 alpha=.7
					   )	

	prefix = '../maps/' + state + '/static/'
	filename = prefix + str(alphaW).replace('.', '_') + '_after.png'
	fig.savefig(filename, bbox_inches='tight', dpi=300)
	stars.remove()

	# make bokeh map
	prefix = '../maps/' + state + '/dynamic/'
	filename =  prefix + str(alphaW).replace('.', '_') + '_after.html'
	df = make_bokeh_map(pcnct_df, 'district_final', palette, filename)

	# include before/after transport cost in resulting DataFrame
	df['cost0'] = cost0
	df['cost_final'] = cost

	# keep track of alphaW used to generate optimal districts
	df['alphaW'] = alphaW

	# keep before/after district-level dataframes together
	df_list.append(df)

	return df_list


def plot_state(geo_df, colors, ax, fig, filename, F_opt=None):
	'''
	Function takes geopandas DataFrame and plots the precincts colored according
	to their congressional district. Saves figure at path "filename."

	INPUTS:
	----------------------------------------------------------------------------
	geo_df: geopandas DataFrame
	district_group: string, column name in geo_df which contains each precinct's 
					congressional district
	ax: matplotlib axis object
	fig: matplotlib figure object, used to save final figure
	filename: string, path to saved figure
	F_opt: np.array, plots the location of district offices if provided
	'''	
	# plot patches, colored according to district
	patches = plot_patches(geo_df.geometry.values, colors, ax, lw=.1)
	
	# plot stars for office locations
	if F_opt is not None:
		for i in range(len(F_opt)):
			ax.scatter(F_opt[i, 0], F_opt[i, 1], color='black',
					   marker='*', s=30, alpha=1
					   )	

	if filename is not None:
		ax.set_yticklabels([])
		ax.set_xticklabels([])
		fig.savefig(filename, bbox_inches='tight', dpi=100)
		plt.close()			

	return patches


def make_bokeh_map(pcnct_df, groupvar, palette, filename):
	'''
	This file makes a map using the Bokeh library and a GeoPandas DataFrame.

	INPUTS:
	----------------------------------------------------------------------------
	pcnct_df: GeoPandas DataFrame, contains columns 'geometry', as well as 
				demographic and political variables used to make a chloropleth
	groupvar: string, column name identifying congressional districts
	filename: string, destination of bokeh plot

	OUTPUTS:	
	----------------------------------------------------------------------------
	df: pandas DataFrame, contrains district-level information
	'''		
	output_file(filename)		

	# aggregate precincts by district
	df = pcnct_df.dissolve(by=groupvar, aggfunc=np.sum)

	# aggregating can create multi-polygons, can't plot in bokeh so unpack those
	df = tpf.unpack_multipolygons(df, impute_vals=False)
	df.geometry = [shapely.geometry.polygon.asPolygon(g.exterior) for g in df.geometry.values]

	# smooth out the district level polygons
	df.geometry = df.geometry.simplify(.007).buffer(0.007)

	# remove bleed (nonempty intersection) resulting from buffer command
	for ig, g in enumerate(df.geometry):
		for ig2, g2 in enumerate(df.geometry):
			if ig != ig2:
				g -= g2
		df.geometry.iloc[ig] = g

	# carry over important variables into the district-level dataframe 
	df['area'] = df.geometry.area	
	df['DEM'] = df[['PRES04_DEM','PRES08_DEM','PRES12_DEM']].mean(axis=1)
	df['REP'] = df[['PRES04_REP','PRES08_REP','PRES12_REP']].mean(axis=1)

	# district level variables
	df['BLACK_PCT'] = (df['POP_BLACK'].values/df['POP_TOTAL'].values)*100
	df['HISPAN_PCT'] = (df['POP_HISPAN'].values/df['POP_TOTAL'].values)*100
	df['REP_PCT'] = (df['REP'].values/df[['REP','DEM']].sum(axis=1).values)*100
	df['DEM_PCT'] = (df['DEM'].values/df[['REP','DEM']].sum(axis=1).values)*100
	df['POP_PCT'] = (df['POP_TOTAL'].values/df['POP_TOTAL'].sum())*100
	df['dist'] = df.index.values.astype(int) + 1
	df['n_precincts'] = len(pcnct_df)

	# variables for mapping
	df['patchx'] = df.geometry.apply(lambda x: tpf.get_coords(x, xcoord=True))
	df['patchy'] = df.geometry.apply(lambda x: tpf.get_coords(x, xcoord=False))
	df['color1'] = [palette[i-1] for i in df.dist]

	source = CDS(data=dict(
							x = df['patchx'].values.astype(list),
							y = df['patchy'].values.astype(list),
							color1 = df['color1'].values.astype(list),
							dist = df['dist'].values.astype(list),
							pop_pct  = df['POP_PCT'].values.astype(list),
							black_pct  = df['BLACK_PCT'].values.astype(list),
							hisp_pct  = df['HISPAN_PCT'].values.astype(list),		
							rep_pct  = df['REP_PCT'].values.astype(list),
							dem_pct  = df['DEM_PCT'].values.astype(list),
							)
			    )

	# adjust image size according to shape of state
	lat_range = pcnct_df.INTPTLAT10.max() - pcnct_df.INTPTLAT10.min()
	lon_range = pcnct_df.INTPTLON10.max() - pcnct_df.INTPTLON10.min()
	
	# make bokeh figure. 
	TOOLS = "pan,wheel_zoom,box_zoom,reset,hover"	
	p = figure(plot_width=420,
			   plot_height=int(420*(lat_range/lon_range)*1.2),
			   tools=TOOLS,
			   toolbar_location='above')

	# hover settings
	hover = p.select_one(HoverTool)
	hover.point_policy = "follow_mouse"
	hover.tooltips = [	("District", "@dist"),
						("Population Share", "@pop_pct%"),
						("Black Pop.", "@black_pct%"),
						("Hispanic Pop.", "@hisp_pct%"),
			            ("Democrat", "@dem_pct%"),
			            ("Republican", "@rep_pct%")]
	
	# remove bokeh logo
	p.toolbar.logo = None
	p.patches('x','y', source=source, 
	          fill_color='color1', fill_alpha=.9, 
	          line_color='black', line_width=.5,
	          line_alpha=.4)

	# Turn off tick labels
	p.axis.major_label_text_font_size = '0pt'  
	
	# Turn off tick marks 	
	p.grid.grid_line_color = None
	p.outline_line_color = "black"
	p.outline_line_width = .5
	p.outline_line_alpha = 1
	# p.background_fill_color = "gray"
	p.background_fill_alpha = .1
	p.axis.major_tick_line_color = None  # turn off major ticks
	p.axis[0].ticker.num_minor_ticks = 0  # turn off minor ticks
	p.axis[1].ticker.num_minor_ticks = 0

	# save output as html file	
	# show(p)
	save(p)

	# return district level DataFrame
	return df.groupby('dist').first()


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
	----------------------------------------------------------------------------
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


def make_palette(n_districts, cmap=plt.cm.Paired, hex=False):
	'''
	Takes matplotlib cmap object and generates a palette of n equidistant points
	over the cmap spectrum, returned as a list. 
	
	INPUTS:
	----------------------------------------------------------------------------
	n_districts: int, number of districts
	cmap: matplotlib colormap object, e.g. cmap = plt.cm.Paired
	hex: boolean (default=False), If true returns colors as hex strings instead 
		 of rgba tuples.

	OUTPUTS:
	----------------------------------------------------------------------------
	palette: list, list of size n_districts of colors
	'''
	# define the colormap
	cmaplist = [cmap(i) for i in range(cmap.N)]		
	palette =[cmaplist[i] for i in range(0, cmap.N, int(cmap.N/n_districts))]	

	if hex is True:
		palette = [mpl.colors.rgb2hex(p) for p in palette]

	return palette


def make_histplot(df_list, state, labels):
	'''
	Takes table output from make_district_df() and plots it. This allows us to 
	have compact results even when there is a large number of districts. 

	INPUTS:
	----------------------------------------------------------------------------
	state: string, key of "states" dict, which pairs states with abbreviations
	alphaW_list: list, list of alphaW parameters (scalars in [0,1])
	file_end: ending of filename for figure	

	OUTPUTS:
	----------------------------------------------------------------------------
	None
	'''	
	sns.set_style(style='white')
	# mpl.rc('font',family='serif')
	tpf.make_folder('../analysis/' + state)
	
	TOOLS = "save"
	p_shell = figure(plot_width=675,
			   plot_height=380,
			   tools=TOOLS,
			   toolbar_location='right')	

	# initialize figure
	titles = ['Approximate Distribution of Republican Voter Shares across Districts',
			'Approximate Distribution of Black Population Shares across Districts',
			'Approximate Distribution of Hispanic Population Shares across Districts']

	xlabels = ['District-Level Republican Voter Shares',
			   'District-Level Black Population Shares',
			   'District-Level Hispanic Population Shares']

	for ivar, var in enumerate(['REP_PCT','BLACK_PCT','HISPAN_PCT']):
		fig, ax = sns.plt.subplots(1, 1, figsize=(13, 5), sharey=True)
		bins = 10
		kde_kws = {"shade": True, 'kernel':'gau', 'bw':.04}

		for i in range(1):
			old_df = df_list[0]
			new_df = df_list[1 + i]			
			sns.distplot(old_df[var]/100, hist=False, bins=bins,
						 kde_kws=kde_kws, ax=ax,
						 label='2010 Districts', norm_hist=True
						 )
			
			sns.distplot(new_df[var]/100, hist=False, bins=bins, 
						 kde_kws=kde_kws, ax=ax, 
						 label='Optimal Districts', norm_hist=True
						 )		
			
			ax.set_xlim(0, 1)
			ax.set_ylim(0, 1)
			ax.set_xlabel(xlabels[ivar], fontsize=12)
			ax.set_ylabel('Number of Districts', fontsize=12)
			ax.legend(loc=0, ncol=1, fontsize=14)

			if i == 0:
				p = bokeh_mpl.to_bokeh()
				p.title.text = titles[ivar]
				p.toolbar.logo = None
				p.tools = p_shell.tools
				p.plot_height = p.plot_height
				p.plot_width = p_shell.plot_width
				p.toolbar_location = p_shell.toolbar_location
				p.xgrid.grid_line_color = None
				p.ygrid.grid_line_color = None

				output_file("../analysis/"+state+"/"+var.lower()+"_kde.html")		
				save(p)	

		# filename = '../analysis/'+state+'/'+var+'_kde.pdf'
		# fig.savefig(filename, bbox_inches='tight', dpi=100)
		plt.close()

	return None

def calculate_EG(df_list):
#df_list comes from make_district_df()
#Assumption at the moment: all the districts have equal population.

        output = []

        for df in df_list:
                wasted_Rep = 0
                wasted_Dem = 0
                pct_vec = df['REP_PCT']/100
                for x in pct_vec:

                        if x < .5:
                                wasted_Rep += x/len(df)
                                wasted_Dem += (.5 - x)/len(df)

                        if x > .5:
                                wasted_Rep += (x-.5)/len(df)
                                wasted_Dem +=(1-x)/len(df)

                output.append(abs(wasted_Rep - wasted_Dem))
        return output


def make_barplot(df_list, state, labels):	
	'''
	Takes table output from make_district_df() and plots it. This allows us to 
	have compact results even when there is a large number of districts. 

	INPUTS:
	----------------------------------------------------------------------------
	state: string, key of "states" dict, which pairs states with abbreviations
	alphaW_list: list, list of alphaW parameters (scalars in [0,1])
	file_end: ending of filename for figure	

	OUTPUTS:
	----------------------------------------------------------------------------
	None

	'''	
	sns.set_style(style='darkgrid')
	mpl.rc('font',family='serif')
	tpf.make_folder('../analysis/' + state)

	n_subplots = len(df_list)
	cmap = mpl.cm.get_cmap('brg')
	fig, ax = sns.plt.subplots(n_subplots, 1, figsize=(7, 5), sharex=True)
	
	for idf, df in enumerate(df_list):
		df.sort_values(by='REP_PCT', inplace=True)
		colors = [cmap(i/2) for i in df['REP_PCT'].values]
		
		x1 = np.arange(len(df))
		x2 = np.arange(len(df) + 1)
		y1 = np.ones(len(df),)
		y2 = np.ones(len(df) + 1,)
		
		dem_bar = df['DEM_PCT'].values/100
		ax[idf].bar(x1, y1, color='r', linewidth=0, width=1.0, alpha=.8)
		ax[idf].bar(x1, dem_bar, color='b', linewidth=0, width=1.0, alpha=.8)
		
		# horizontile line at .5
		ax[idf].plot(x2, y2*.5, color='w', linewidth=.2, alpha=.8)

		ax[idf].set_xticklabels([])
		ax[idf].set_xlim(0, len(df))
		ax[idf].set_ylim(0, 1)
		ax[idf].set_ylabel(labels[idf])		

	filename = '../analysis/' + state + '/barplot.pdf'
	fig.savefig(filename, bbox_inches='tight', dpi=100)
	plt.close()

	return None


states = {
			'AL': 'Alabama',
			# 'AK': 'Alaska',
			# 'AZ': 'Arizona',TODO
			# 'AR': 'Arkansas',
			'CA': 'California',
			'CO': 'Colorado',
			'CT': 'Connecticut',
			'FL': 'Florida',
			'GA': 'Georgia',
			# 'HI': 'Hawaii', #problem here
			'ID': 'Idaho',
			'IL': 'Illinois',
			'IN': 'Indiana',
			'IA': 'Iowa',
			'KS': 'Kansas',
			'KY': 'Kentucky', 
			'LA': 'Louisiana',
			'ME': 'Maine',
			'MD': 'Maryland',
			'MA': 'Massachusetts',
			'MI': 'Michigan',
			'MN': 'Minnesota',
			'MS': 'Mississippi',
			'MO': 'Missouri',
			# 'MT': 'Montana',
			'NE': 'Nebraska',
			'NV': 'Nevada', 
			'NH': 'New Hampshire',
			'NJ': 'New Jersey',
			'NM': 'New Mexico',
			'NY': 'New York',
			'NC': 'North Carolina',
			# 'ND': 'North Dakota',
			'OH': 'Ohio',
			'OK': 'Oklahoma',
			'OR': 'Oregon',
			'PA': 'Pennsylvania',
			'RI': 'Rhode Island',
			'SC': 'South Carolina',
			# 'SD': 'South Dakota',
			'TN': 'Tennessee',
			'TX': 'Texas',
			'UT': 'Utah',
			# 'VT': 'Vermont',
			'VA': 'Virginia',
			'WA': 'Washington',
			'WV': 'West Virginia',
			'WI': 'Wisconsin',
			# 'WY':'Wyoming'
			}


if __name__ == '__main__':

	cost_df = pd.DataFrame()
	state_list = list(states.keys())
	state_list.sort()

	hist_labels = ['Current', r"$\alpha_W=0$", r"$\alpha_W=.25$", r"$\alpha_W=.75$"]

	state_df_list = []
	for state in state_list:
		# get data from shapefiles if not available already
		# get_state_data(state)

		# make maps
		df_list = make_state_maps(state)

		# make charts for each state
		make_histplot(df_list[0:3], state, hist_labels)
		make_barplot(df_list[0:3], state, hist_labels)

		# save results (list of tuples)
		state_df_list.append((state, df_list))

	# pickle results for later
	pickle.dump(state_df_list, open('../analysis/state_dfs.p', 'wb'), protocol=2) 
