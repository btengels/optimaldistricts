from __future__ import division

import numpy as np

import pandas as pd
import geopandas as geo
import pickle
import shapely

import time
import transport_plan_functions as tpf
import sklearn.metrics as metrics

from matplotlib import pyplot as plt
import seaborn as sns

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


# TODO: make some tables w/ some other variables

final_tables = pd.read_pickle('../tables/before_after_stats_nodemographics.p')
final_tables.sort_values(by='cost_improve', inplace=True, ascending=False)


most_least_gerryed  = pd.DataFrame( final_tables.cost_improve.head(5))
most_least_gerryed['least_states'] =  final_tables.cost_improve.tail(5).index.values
most_least_gerryed['least_cost'] = final_tables.cost_improve.tail(5).values
# least_gerryed = final_tables.cost_improve.tail(5)
demographics_new = final_tables[['new_maxblack','new_meanblack','new_medblack','new_minblack']]
demographics_old = final_tables[['old_maxblack','old_meanblack','old_medblack','old_minblack']]

demographics_new.index = ['new' + d for d in demographics_new.index]
demographics_new.columns = ['maxblack','meanblack','medblack','minblack']

demographics_old.index = ['old' + d for d in demographics_old.index]
demographics_old.columns = ['maxblack','meanblack','medblack','minblack']


print most_least_gerryed.to_latex()

print demographics_old.to_latex()
print demographics_new.to_latex()
