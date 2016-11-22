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
	for ivar, var in enumerate(['rep_pct','blk_pct','hisp_pct']):
		fig, ax = sns.plt.subplots(1, 2, figsize=(14, 5), sharey=True)
		bins = 10
		kde_kws = {"shade": True, 'kernel':'gau', 'bw':.05}

		for i in range(2):
			old_df = df_list[0]
			new_df = df_list[1 + i]			
			sns.distplot(old_df[var]/100, hist=False, bins=bins, kde_kws=kde_kws, ax=ax[i], label=labels[0])			
			sns.distplot(new_df[var]/100, hist=False, bins=bins, kde_kws=kde_kws, ax=ax[i], label=labels[1 + i])					
			
			ax[i].set_xlim(0, 1)
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
		df.sort_values(by='rep_pct', inplace=True)
		colors = [cmap(i/2) for i in df['rep_pct'].values]
		
		x1 = np.arange(len(df))
		x2 = np.arange(len(df) + 1)
		y1 = np.ones(len(df),)
		y2 = np.ones(len(df) + 1,)
		
		ax[idf].bar(x1, y1, color='r', linewidth=0, width=1.0, alpha=.8)
		ax[idf].bar(x1, df['dem_pct'].values/100, color='b', linewidth=0, width=1.0, alpha=.8)
		
		# horizontile line at .5
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
			# 'ME':'Maine',
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
		df = pd.read_pickle('../tables/'+ state +'/results_before.p')
		state_df_list.append( df )		

		for alphaW in alphaW_list:
			df = pd.read_pickle('../tables/'+ state +'/results_'+str(alphaW)+'.p')	
			state_df_list.append( df )
	
		# make figures from list of dataframes
		titles = ['Current',r"$\alpha_W=0$",r"$\alpha_W=.25$", r"$\alpha_W=.75$"]
		make_histplot( state_df_list, state, titles)
		make_barplot( state_df_list, state, titles)

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
	make_histplot( combined_df_list, 'US', titles)
	make_barplot( combined_df_list, 'US', titles)
