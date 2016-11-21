#optimaldistricts

This repo contains code/data to impliment the districting algorithm used the paper "A Political Redistricting Algorithm with Communities of Interest" as well as the website [optimaldistricts.org](http://optimaldistricts.org). The algorithm draws district boundaries in order to minimize transport distances between constituents and representatives while maintaining, as much as possible, demographic communities of interest. 

The districting problem is difficult from a computational standpoint (NP-complete, actually). Composing an optimal set of districts from larger geographic units (census tracts or precincts) simplifies the problem somewhat, but many states have several thousands of precincts. Requirements such as nearly-equal population shares across districts, geographic contiguity, or generating a sufficient number of majority-minority districts, complicates the problem further. 

## Code and Data
The precinct-level data are first taken from [autoredistrict.org](http://autoredistrict.org/). The raw data are comprised of shapefiles (.shp) which we then pull into [GeoPandas](http://geopandas.org/) for our analysis. The data include demographics and presidential election results at the precinct level for all 50 states. 

The code is written in Python and aims to be tidy/readable so that anyone can use and adapt the code for their respective state. Currently, there are three important scripts used to generate the maps/graphs/etc. for each state.

- **get_data.py**: This script is prepares the Geopandas DataFrames, which are saved alongside the shapefiles. This file drops adjacent bodies of water, drops all unnecessary variables, and reduces the complexity of the polygons so our maps can load quickly online. If you downloaded the Data-Files folder and see some pickled (.p) files, you don't need to run this script unless you want to change the DataFrames in some way. 

- **transport\_plan_functions.py**: This file contains functions used in computing the optimal districts. The primary function is  `gradientDescentOT()`, and most of the other functions in the file exist to support this main function. All of the functions have detailed doc strings describing their inputs/outputs and functionality. 

- **make_maps.py**: This script reads in the pickled DataFrames and for each state with 2 or more congressional districts...
  * Makes a map of the current (2010) congresisonal districts (approximate since we keep precincts intact). Two kinds of maps are made, a static .png and then a web-ready [bokeh](http://bokeh.pydata.org/en/latest/) plot.
  *  Computes the optimal districts for a given value of alphaW.
  *  (beta) Checks to see if the resulting map is comprised of contiguous districts. If the districts are contiguous, the script computes new districts with highest possible value for alphaW that still respects contiguity. 
  *  Makes .png and html maps of the new districts.
  *  Plots kernel density plots of district-level statistics
  *  Saves a list of dataframes used in making the maps...to be used in ongoing projects (ranking states by a gerrymandering index, looking at the number of competitive seats, etc.)

 
Authors:
+ Benjamin Tengelsen
+ Ryan Murray