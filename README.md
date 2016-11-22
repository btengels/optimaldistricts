#Optimal Districts
This repo contains code/data to impliment the districting algorithm used the paper "A Political Redistricting Algorithm with Communities of Interest" as well as the website [optimaldistricts.org](http://optimaldistricts.org). The algorithm draws district boundaries in order to minimize the distance between constituents and representatives while maintaining, as much as possible, demographic communities of interest. 

The districting problem is difficult from a computational standpoint (NP-complete, actually). Composing an optimal set of districts from larger geographic units (census tracts or precincts) simplifies the problem somewhat, but many states have several thousand precincts which is enough to make the problem slow using standard software/techniques. Requirements such as nearly-equal population shares across districts, geographic contiguity, or generating a sufficient number of majority-minority districts, complicate the problem further. 

We overcome this computational hurdle using some recent numerical tools from the Optimal Transport community (hence the name "optimal" districts). Given the location of district offices, we use a regularized penalty function and an iterative projection algorithm to quickly assign precincts to districts (Sinkhorn algorithm). Given this (new) set of districts, we find the best location for district offices, then repeat until convergence (Lloyd's algorithm/constrained-kmeans algorithm).

## Code and Data
The precinct-level data are first taken from [autoredistrict.org](http://autoredistrict.org/). The raw data are comprised of shapefiles (.shp) which we then pull into [GeoPandas](http://geopandas.org/) for our analysis. The data include demographics and presidential election results at the precinct level for all 50 states. 

The code is written in Python and aims to be tidy/readable and simple as possible so that anyone can use and adapt the code as they like. Currently, there are two important scripts used to generate the maps/graphs/etc. for each state.

- **transport\_plan_functions.py**: This file contains functions used in computing the optimal districts. The primary function is  `gradientDescentOT()`, and most of the other functions in the file exist to support this main function. All of the functions have detailed doc strings describing their inputs/outputs and functionality. 

- **make_maps.py**: 
  * Reads in the shapefiles and simplifies the precinct polygons for faster rendering, stores reuslts as a pickled Geopandas DataFrame
  * Reads in the pickled DataFrames and for each state with 2 or more congressional districts...
  * Makes a map of the current (2010) congresisonal districts (approximate since we keep precincts intact). Two kinds of maps are made, a static .png and a web-ready [bokeh](http://bokeh.pydata.org/en/latest/) plot.
  *  Computes the optimal districts for a given value of alphaW.
  *  (beta) Checks to see if the resulting map is comprised of contiguous districts. If the districts are contiguous, the script computes new districts with highest possible value for alphaW that still respects contiguity. 
  *  Makes .png and html maps of the new districts.
  *  Plots kernel density plots of district-level statistics
  *  Saves a list of dataframes used in making the maps...to be used in ongoing projects (ranking states by a gerrymandering index, looking at the number of competitive seats, etc.)

## A work in progress
We have a long list of additional features to add and questions to answer using our algorithm. Some of our short-term TODOs include:
1. parallelize everything
1. improve/speed up checks for contiguity
1. optimally determine the regularization and demographic distance parameters
1. measure "wasted" voter shares
1. generalize functions in `transport\_plan_functions.py` so that an arbitrary geopandas DataFrame with the necessary lat/lon variables and some categorical column can solve the districting problem (sometimes called the "warehousing problem").
 
Thanks for reading and check back for updates!
