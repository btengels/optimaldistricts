from __future__ import division

import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import os

import matplotlib as mpl
mpl.rc('font', family='serif')

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


def make_appendix_page(state, table_tex):
	'''
	This function writes the appendix to our paper as a latex file. Each state has 2 pages.
	The first page includes maps, while the second includes figures with district stats.

	INPUTS:
	-------------------------------------------------------------------------------
	state: string, key of dictionary "states" which pairs states with abbreviations
	table_tex: string, tex output from df.to_latex(), not currently in use.

	OUTPUTS:
	-------------------------------------------------------------------------------
	None

	'''	
	# subsection header
	f.write(r'\subsection{' + states[state] + '}'); f.write('\n')

	# Page 1: Maps
	f.write(r'\begin{figure}[htb!] \centering'); f.write('\n')	
	f.write(r'\caption{ Current Districts }'); f.write('\n')	
	f.write(r'\includegraphics[width=5in,height=3in,keepaspectratio]{../maps/'+  state +'/static/0_before.png}') ; f.write('\n')	
	f.write(r'\caption{ New Districts ($\alpha_w$ = 0) }'); f.write('\n')	
	f.write(r'\includegraphics[width=5in,height=3in,keepaspectratio]{../maps/'+  state +'/static/0_after.png}') ; f.write('\n')
	f.write(r'\caption{ New Districts ($\alpha_w$ = 1) }'); f.write('\n')	
	f.write(r'\includegraphics[width=5in,height=3in,keepaspectratio]{../maps/'+  state +'/static/1_after.png}') ; f.write('\n')
	f.write(r'\end{figure}'); f.write('\n')
	f.write('\n')

	f.write(r'\clearpage') ; f.write('\n')
	f.write(r'\newpage') ; f.write('\n')
	f.write('\n')	

	# Page 2: figures w/ district statistics 
	f.write(r'\begin{figure}[htb!] \centering'); f.write('\n')	
	f.write(r'\caption{ Demographics: black population }'); f.write('\n')	
	f.write(r'\includegraphics[width=4.5in]{../analysis/'+state+'/analysis_scatter.png}') ; f.write('\n')	
	f.write(r'\caption{ Politics: democratic population (placeholder)}'); f.write('\n')	
	f.write(r'\includegraphics[width=4.5in]{../analysis/'+state+'/analysis_scatter2.png}') ; f.write('\n')	
	f.write(r'\end{figure}'); f.write('\n')
	f.write('\n')

	# table
	# for t in table_tex.split('\n'):
	# 	f.write(t)
	# 	f.write(' \n')
	# 
	f.write(r'\clearpage') ; f.write('\n')
	f.write(r'\newpage') ; f.write('\n')
	f.write('\n')	
	
	return None


def make_table1(state, alphaW, groupby_var):
	'''
	Makes a table listing key statistics of congressional districts, including
	the share of the total state population, the share of black population within
	each district, and the share of republican/democrat votes within each district.
	
	INPUTS:
	-------------------------------------------------------------------------------
	state: string, key of "states" dict, which pairs states with abbreviations
	alphaW: scalar in [0,1]
	groupby_var: string, column of DataFrame indicating the column to groupby over
	
	OUTPUTS:
	-------------------------------------------------------------------------------	
	df: pandas DataFrame, abbreviated dataframe with districts (rows) and a few key
		statistics (columns). 
	'''
	# read in data, take sum over groupby_var
	table_df = pd.read_pickle('../tables/'+ state +'/results_'+str(alphaW)+'.p')
	df = table_df.groupby(groupby_var).agg(np.sum)

	# add new df columns
	key_vars = ['pop_pct_', 'black_pct_', 'hisp_pct_', 'R_pct_', 'D_pct_']
	names = [k + str(alphaW) for k in key_vars]

	df[ names[0] ] = df.POP_TOTAL/df.POP_TOTAL.sum()
	df[ names[1] ] = df.POP_BLACK/df.POP_TOTAL
	df[ names[2] ] = df.POP_HISPAN/df.POP_TOTAL
	df[ names[3] ] = df['CD12_REP']/df[['CD12_REP','CD12_DEM']].sum(axis=1)
	df[ names[4] ] = df['CD12_DEM']/df[['CD12_REP','CD12_DEM']].sum(axis=1)

	return df[names]


def make_analysis_figs(state, alphaW_list, file_end):	
	'''
	Takes table output from make_table1() and plots it. This allows us to 
	have compact results even when there is a large number of districts. 

	INPUTS:
	-------------------------------------------------------------------------------
	state: string, key of "states" dict, which pairs states with abbreviations
	alphaW_list: list, list of alphaW parameters (scalars in [0,1])
	file_end: ending of filename for figure	

	OUTPUTS:
	-------------------------------------------------------------------------------
	None

	'''	
	# make district states for each level of alphaW in alphaW_list
	old_df = make_table1(state,  alphaW_list[0], 'CD_2010')
	new_df1 = make_table1(state, alphaW_list[1], 'district_iter19')
	new_df2 = make_table1(state, alphaW_list[2], 'district_iter19')		

	# initialize figure
	fig, ax = sns.plt.subplots(1,2, figsize=(10,5), sharey=True)

	# make left and right plots (1x2 subplot)
	for ix, xaxis,yaxis in zip([0,1],['pop_pct_','R_pct_'], ['black_pct_','hisp_pct_']):
		ax[ix].scatter(old_df[xaxis+str(alphaW_list[0])], old_df[yaxis+str(alphaW_list[0])], color=palette[0], marker='o', alpha=.8, s=80, label='Current Districts')
		ax[ix].scatter(new_df1[xaxis+str(alphaW_list[1])], new_df1[yaxis+str(alphaW_list[1])], color=palette[1], marker='*', alpha=.8, s=80, label='New Districts: '+str(alphaW_list[1]))
		ax[ix].scatter(new_df2[xaxis+str(alphaW_list[2])], new_df2[yaxis+str(alphaW_list[2])], color=palette[2], marker='^', alpha=.8, s=80, label='New Districts: '+str(alphaW_list[2]))

	# left plot: scatter of total population (xaxis) and black_pct (yaxis)
	ax[0].set_xlabel('Percent of Total Population',fontsize=13)	
	ax[0].set_ylabel('Black Percent of District Population',fontsize=13)

	# right plot: scatter of Republican votes (xaxis) and hispanic_percent (yaxis)
	ax[1].set_xlabel('Republican Percent', fontsize=13)	
	ax[1].set_ylabel('Hispanic Percent of District Population',fontsize=13)
	ax[1].set_xlim(0,1)
	ax[1].legend(bbox_to_anchor=(1.6, .75), fontsize=13) #legend on the bottom
	
	# plt.show()
	fig.tight_layout()
	fig.savefig('../analysis/'+state+'/'+file_end, bbox_inches='tight', dpi=100)
	plt.close()	

	return None


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

	# set colors for statistical plots 
	palette = sns.color_palette('colorblind', n_colors=3)

	# open appendix tex file
	f = open( '../writeup/state_by_state_appendix.tex', 'w')	
	
	# values of alphaW to include in figures
	alphaW_list1 = [0.0, 0.0, 0.1]
	alphaW_list2 = [0.0, 0.2, 0.3]

	# make mapes and figures for each state
	for state in states:
		print state
		make_folder('../analysis/'+state)
		make_analysis_figs( state, alphaW_list1, 'analysis_scatter.pdf')
		make_analysis_figs( state, alphaW_list2, 'analysis_scatter2.pdf')

		# write these figures to appendix file for each state
		make_appendix_page(state, [])
     
	# close tex file
	f.close()
