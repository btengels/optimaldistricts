from __future__ import division

import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import os 

import matplotlib as mpl
plt.rc('text', usetex=True)
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


def make_appendix_page(state):
	'''
	This function writes the appendix to our paper as a latex file. Each state has 2 pages.
	The first page includes maps, while the second includes figures with district stats.

	INPUTS:
	-------------------------------------------------------------------------------
	state: string, key of dictionary "states" which pairs states with abbreviations

	OUTPUTS:
	-------------------------------------------------------------------------------
	None
	'''	
	# subsection header
	f.write(r'\subsection{' + states[state] + '}'); f.write('\n')

	if state in ['AL','AZ','CA','FL','GA','ID','IL','IN','LA','MI','MO','MS','NJ','NM','OH','TX','UT','SC','WI','WV']:
		width = str(4.5)
		height = str(2.5)

	elif state in ['AR','CO','MN','NY','OR','PA']:
		width = str(3.5)
		height = str(3.5)

	elif state in ['NC','KS','OK','TN','VA','CT','IA','MA','MD','WA']:
		width = str(6)
		height = str(4)

	else:
		width = str(5)
		height = str(3)
	


	# Page 1: Maps
	f.write(r'\begin{figure}[htb!] \centering'); f.write('\n')	
	f.write(r'\caption{ Current Districts }'); f.write('\n')	
	f.write(r'\includegraphics[width='+width+'in,height='+height+'in,keepaspectratio]{../maps/'+  state +'/static/before.png}') ; f.write('\n')	
	f.write(r'\caption{ New Districts ($\alpha_w=0$) }'); f.write('\n')	
	f.write(r'\includegraphics[width='+width+'in,height='+height+'in,keepaspectratio]{../maps/'+  state +'/static/0_0_after.png}') ; f.write('\n')
	f.write(r'\caption{ New Districts ($\alpha_w=.25$) }'); f.write('\n')	
	f.write(r'\includegraphics[width='+width+'in,height='+height+'in,keepaspectratio]{../maps/'+  state +'/static/0_25_after.png}') ; f.write('\n')
	f.write(r'\end{figure}'); f.write('\n')
	f.write('\n')

	f.write(r'\clearpage') ; f.write('\n')
	f.write(r'\newpage') ; f.write('\n')
	f.write('\n')	

	# Page 2: figures w/ district statistics 
	f.write(r'\begin{figure}[htb!] \centering'); f.write('\n')	
	f.write(r'\caption{ Politics: democratic population (placeholder)}'); f.write('\n')	
	f.write(r'\includegraphics[width=7in]{../analysis/'+state+'/R_pct_kde.pdf}') ; f.write('\n')	
	f.write(r'\caption{ Demographics: black population }'); f.write('\n')	
	f.write(r'\includegraphics[width=7in]{../analysis/'+state+'/black_pct_kde.pdf}') ; f.write('\n')	
	f.write(r'\caption{ Demographics: hispanic population }'); f.write('\n')	
	f.write(r'\includegraphics[width=7in]{../analysis/'+state+'/hisp_pct_kde.pdf}') ; f.write('\n')	
	f.write(r'\end{figure}'); f.write('\n')
	f.write('\n')


	f.write(r'\clearpage') ; f.write('\n')
	f.write(r'\newpage') ; f.write('\n')
	f.write('\n')	

	f.write(r'\begin{figure}[htb!] \centering'); f.write('\n')	
	f.write(r'\caption{ Politics: democratic population (placeholder)}'); f.write('\n')	
	f.write(r'\includegraphics[width=6.5in]{../analysis/'+state+'/barplot.pdf}') ; f.write('\n')
	f.write(r'\end{figure}'); f.write('\n')
	f.write('\n')

	f.write(r'\clearpage') ; f.write('\n')
	f.write(r'\newpage') ; f.write('\n')
	f.write('\n')	
	
	return None


def make_district_df(state, alphaW, groupby_var):
	'''
	Makes a df listing key statistics of congressional districts, including
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

	# 
	df['REP_avg'] = df[['PRES04_REP','PRES08_REP','PRES12_REP']].mean(axis=1)
	df['DEM_avg'] = df[['PRES04_DEM','PRES08_DEM','PRES12_DEM']].mean(axis=1)

	# add new df columns
	names = ['pop_pct', 'black_pct', 'hisp_pct', 'R_pct', 'D_pct']	

	df[ names[0] ] = df.POP_TOTAL/df.POP_TOTAL.sum()
	df[ names[1] ] = df.POP_BLACK/df.POP_TOTAL
	df[ names[2] ] = df.POP_HISPAN/df.POP_TOTAL
	df[ names[3] ] = df['REP_avg']/df[['REP_avg','DEM_avg']].sum(axis=1)
	df[ names[4] ] = df['DEM_avg']/df[['REP_avg','DEM_avg']].sum(axis=1)
	df[ 'vote_result' ] = 0
	df.loc[df['D_pct']>.6, 'vote_result'] = 1
	df.loc[df['D_pct']<.4, 'vote_result'] = 2

	vars = names.append('vote_result')
	return df[names]


def make_histplot(df_list, state, labels):	
	'''
	Takes table output from make_district_df() and plots it. This allows us to 
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
	sns.set_style(style='darkgrid')
	mpl.rc('font',family='serif')
	
	# initialize figure
	xlabels = ['Share of Republican Votes','Black Population (percent)','Hispanic Population (percent)']
	for ivar, var in enumerate(['R_pct','black_pct','hisp_pct']):
		fig, ax = sns.plt.subplots(1,2, figsize=(14,5), sharey=True)
		bins = 10
		kde_kws = {"shade": True, 'kernel':'gau','bw':.05,'clip':(0,1)}

		for i in range(2):
			old_df = df_list[0]
			new_df = df_list[1+i]			
			sns.distplot(old_df[var], hist=False, bins=bins, kde_kws=kde_kws, ax=ax[i], label=labels[0])			
			sns.distplot(new_df[var], hist=False, bins=bins, kde_kws=kde_kws, ax=ax[i], label=labels[1+i])					
			
			ax[i].set_xlim(0,1)
			ax[i].set_xlabel(xlabels[ivar], fontsize=14)
			ax[0].set_ylabel('Number of Districts', fontsize=14)
			ax[i].legend(bbox_to_anchor=(.78, -.15), ncol=2, fontsize=14)
		
		fig.tight_layout()
		fig.savefig('../analysis/'+state+'/'+var+'_kde.pdf', bbox_inches='tight', dpi=100)
		plt.close()
	return None



def make_barplot(df_list, state, labels):	
	'''
	Takes table output from make_district_df() and plots it. This allows us to 
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

	n_subplots = len(df_list)
	cmap = mpl.cm.get_cmap('brg')
	fig, ax = sns.plt.subplots(n_subplots, 1, figsize=(7, 5), sharex=True)
	
	for idf, df in enumerate(df_list):
		df.sort_values(by='R_pct', inplace=True)
		colors = [cmap(i/2) for i in df['R_pct'].values]
		
		x1 = np.arange(len(df))
		x2 = np.arange(len(df) + 1)
		y1 = np.ones(len(df),)
		y2 = np.ones(len(df) + 1,)
		
		ax[idf].bar(x1, y1, color='r', linewidth=0, width=1.0, alpha=.8)
		ax[idf].bar(x1, df['D_pct'].values, color='b', linewidth=0, width=1.0, alpha=.8)
		ax[idf].plot(x2, y2*.5, color='w', linewidth=.2, alpha=.8)

		ax[idf].set_xticklabels([])
		ax[idf].set_xlim(0, len(df))
		ax[idf].set_ylim(0, 1)
		ax[idf].set_ylabel(labels[idf])		

	fig.savefig('../analysis/' + state + '/barplot.pdf', bbox_inches='tight', dpi=100)
	plt.close()
	return None


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

	# set colors for statistical plots 
	palette = sns.color_palette('colorblind', n_colors=3)

	# open appendix tex file
	f = open( '../writeup/state_by_state_appendix.tex', 'w')	
	
	# values of alphaW to include in figures
	alphaW_list = [0.0, 0.25, 0.75]

	# make mapes and figures for each state (in alphabetical order)
	state_list = list(states.keys())
	state_list.sort()

	# keep track of all state dataframes based on their parameterization
	US_df_list = [[],[],[],[]]

	for state in state_list:
		print(state)
		make_folder('../analysis/' + state)

		# make a list of dataframes, one for the current boundaries and one for each value of alpha
		state_df_list = []
		state_df_list.append( make_district_df(state, 0.0, 'CD_2010') )		

		for alphaW in alphaW_list:
			state_df_list.append( make_district_df(state, alphaW, 'district_iter19') )
	
		# make figures from list of dataframes
		make_histplot( state_df_list, state, ['Current',r"$\alpha_W=0$",r"$\alpha_W=.25$", r"$\alpha_W=.75$"])
		make_barplot( state_df_list, state, ['Current',r"$\alpha_W=0$",r"$\alpha_W=.25$", r"$\alpha_W=.75$"])

		# write page in appendix file for each state
		make_appendix_page(state)

		# save for national results (insert into list of lists)
		for idf, df in enumerate(state_df_list):
			US_df_list[idf].append(df)
     
	# close tex file
	f.close()


	combined_df_list = []
	# national df - current
	for df_list in US_df_list:
		US_df = df_list[0]
		for df in df_list[1:]:
			US_df = US_df.append(df,ignore_index=True)	

		combined_df_list.append(US_df)

	make_folder('../analysis/US')
	make_histplot( combined_df_list, 'US', ['Current',r"$\alpha_W=0$",r"$\alpha_W=.25$", r"$\alpha_W=.75$"])
	make_barplot( combined_df_list, 'US', ['Current',r"$\alpha_W=0$",r"$\alpha_W=.25$", r"$\alpha_W=.75$"])
