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
	I am too lazy to make 50 folders + subfolders, so 
	'''
	try: 
	    os.makedirs(path)
	except OSError:
	    if not os.path.isdir(path):
	        raise


def make_appendix_page(state, table_tex):
	'''
	'''	
	# subsection header	
	f.write(r'\subsection{' + states[state] + '}'); f.write('\n')

	# Maps
	f.write(r'\begin{figure}[htb!] \centering'); f.write('\n')	
	f.write(r'\caption{ Current Districts }'); f.write('\n')	
	f.write(r'\includegraphics[width=5in,height=3in,keepaspectratio]{../maps/'+  state +'/static/0_before.png}') ; f.write('\n')	
	f.write(r'\caption{ New Districts (Black Param = 0) }'); f.write('\n')	
	f.write(r'\includegraphics[width=5in,height=3in,keepaspectratio]{../maps/'+  state +'/static/0_after.png}') ; f.write('\n')
	f.write(r'\caption{ New Districts (Black Param = 1) }'); f.write('\n')	
	f.write(r'\includegraphics[width=5in,height=3in,keepaspectratio]{../maps/'+  state +'/static/1_after.png}') ; f.write('\n')
	f.write(r'\end{figure}'); f.write('\n')
	f.write('\n')

	f.write(r'\clearpage') ; f.write('\n')
	f.write(r'\newpage') ; f.write('\n')
	f.write('\n')	

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
	


def make_table1(state, black_param, groupby_var):
	'''
	'''
	# read in data, take sum over groupby_var
	table_df = pd.read_pickle('../tables/results_'+ state +'_'+str(black_param)+'.p')
	df = table_df.groupby(groupby_var).agg(np.sum)

	# add new df columns
	key_vars = ['pop_pct_', 'black_pct_', 'R_pct_', 'D_pct_']
	names = [k + str(black_param) for k in key_vars]

	df[ names[0] ] = df.total_pop/df.total_pop.sum()
	df[ names[1] ] = df.Black/df.total_pop			
	df[ names[2] ] = df.house_R/df[['house_R','house_D']].sum(axis=1)
	df[ names[3] ] = df.house_D/df[['house_R','house_D']].sum(axis=1)

	return df[names]


def make_analysis_figs( state, blk_prms, file_end):
	'''
	'''
	print blk_prms
	old_df = make_table1(state,  blk_prms[0], 'current_district')
	new_df1 = make_table1(state, blk_prms[1], 'district_iter19')
	new_df2 = make_table1(state, blk_prms[2], 'district_iter19')		

	fig, ax = sns.plt.subplots(1,2, figsize=(10,5),sharey=True)

	ax[0].scatter(old_df['pop_pct_'+str(blk_prms[0])], old_df['black_pct_'+str(blk_prms[0])], color=palette[0], marker='o', alpha=.8, s=80, label='Current Districts')
	ax[0].scatter(new_df1['pop_pct_'+str(blk_prms[1])], new_df1['black_pct_'+str(blk_prms[1])], color=palette[1], marker='*', alpha=.8, s=80, label='New Districts: '+str(blk_prms[1]))
	ax[0].scatter(new_df2['pop_pct_'+str(blk_prms[2])], new_df2['black_pct_'+str(blk_prms[2])], color=palette[2], marker='^', alpha=.8, s=80, label='New Districts: '+str(blk_prms[2]))

	ax[1].scatter(old_df['R_pct_'+str(blk_prms[0])], old_df['black_pct_'+str(blk_prms[0])], color=palette[0], marker='o', alpha=.8, s=80, label='Current Districts')
	ax[1].scatter(new_df1['R_pct_'+str(blk_prms[1])], new_df1['black_pct_'+str(blk_prms[1])], color=palette[1], marker='*', alpha=.8, s=80, label='New Districts: '+str(blk_prms[1]))
	ax[1].scatter(new_df2['R_pct_'+str(blk_prms[2])], new_df2['black_pct_'+str(blk_prms[2])], color=palette[2], marker='^', alpha=.8, s=80, label='New Districts: '+str(blk_prms[2]))	

	# plot settings
	# ax.set_ylim(0,.8)
	 
	ax[0].set_xlabel('Percent of Total Population',fontsize=13)
	ax[1].set_xlabel('Republican Percent of House Votes',fontsize=13)	
	ax[0].set_ylabel('Black Percent of District Population',fontsize=13)
	# ax[1].set_title(state)
	ax[1].set_xlim(0,1)
	ax[1].legend(bbox_to_anchor=(1.6, .75), fontsize=13) #legend on the bottom
	
	# plt.show()
	fig.tight_layout()
	fig.savefig('../analysis/'+state+'/'+file_end, bbox_inches='tight', dpi=100)
	plt.close()	


states = {	'AL':'Alabama',
			'AZ':'Arizona',
			# 'CA':'California',
			'CO':'Colorado', 
			'CT':'Connecticut',  
			'FL':'Florida', 
			# 'GA':'Georgia', 
			# # 'HI':'Hawaii', 
			# 'IA':'Iowa', 
			# # 'ID':'Idaho', 
			# 'IL':'Illinois', 
			# 'IN':'Indiana', 
			# 'KS':'Kansas', 
			# 'LA':'Louisiana', 
			# 'MA':'Massachusetts', 
			# 'MD':'Maryland', 
			# # 'MI':'Michigan', 
			# 'MN':'Minnesota', 
			# 'MO':'Missouri', 
			# 'MS':'Mississippi', 
			# 'NC':'North Carolina', 
			# # 'ND':'North Dakota', 
			# # 'NE':'Nebraska', 
			# # 'NH':'New Hampshire', 
			# 'NJ':'New Jersey', 
			# # 'NM':'New Mexico', 
			# # 'NV':'Nevada', 
			# # 'NY':'New York',
			# 'OH':'Ohio', 
			# 'OK':'Oklahoma', 
			# # 'OR':'Oregon', #problem here
			# 'PA':'Pennsylvania', 
			# # 'RI':'Rhode Island', #problem here
			# 'SC':'South Carolina', 
			# # 'SD':'South Dakota', 
			# 'TN':'Tennessee', 
			# # 'TX':'Texas', 
			# # 'UT':'Utah', #problem here
			# 'VA':'Virginia',# used https://github.com/vapublicaccessproject/va-precinct-maps instead
			# # 'VT':'Vermont', 
			# 'WA':'Washington', 
			# # 'WI':'Wisconsin',
			# # 'WY':'Wyoming'
			}


if __name__ == '__main__':
	palette = sns.color_palette('colorblind', n_colors=3)

	# f = open( '../writeup/state_by_state_appendix.tex', 'w')	
	
	# [0,.1,.3,.5,.75,1,1.5,2.5]
	black_param_list1 = [0, 0, .3]
	black_param_list2 = [0, .75, 1]
	for state in states:
		# df = make_table1(state, 0, 'district_iter19')
		# stop
		make_folder('../analysis/'+state)
		make_analysis_figs( state, black_param_list1, 'analysis_scatter.pdf')
		make_analysis_figs( state, black_param_list2, 'analysis_scatter2.pdf')

		
			# stop
			# cost0 = table_df.cost0.values[0]
			# cost  = table_df.cost_final.values[0]
			
			# #$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
			# # this is where you decide what stats will appear in the tables...
			# state_df[state] = {	'cost_improve': (cost0-cost)/cost0,
			# 					'old_minblack': np.min(old_df.Black.values/old_df.total_pop.values),
			# 					'old_medblack': np.median(old_df.Black.values/old_df.total_pop.values),
			# 					'old_meanblack': np.median(old_df.Black.values/old_df.total_pop.values),
			# 					'old_maxblack': np.max(old_df.Black.values/old_df.total_pop.values),
			# 					'new_minblack': np.min(new_df.Black.values/new_df.total_pop.values),
			# 					'new_medblack': np.median(new_df.Black.values/new_df.total_pop.values),
			# 					'new_meanblack': np.median(new_df.Black.values/new_df.total_pop.values),
			# 					'new_maxblack': np.max(new_df.Black.values/new_df.total_pop.values),	
			# 					#num_Dvotes:  
			# 					 }
		
		# statelist.pop()
		
		# make_appendix_page(state, [])
	
	# close tex file
	# f.close()
