Texas Redistricting Data
Contact: Maxwell Palmer (mpalmer@fas.harvard.edu)
----------------------------------------

This dataset is based upon data provided by the Texas Legislative Council (http://www.tlc.state.tx.us/redist/redist.html).

The TLC data can be downloaded from
ftp://ftpgis1.tlc.state.tx.us/2011_Redistricting_Data

For information on how the TLC constructs their database, read "Data for 2011 Redistricting in Texas", located at http://www.tlc.state.tx.us/redist/pdf/Data_2011_Redistricting.pdf.

----------------------------------------

This dataset includes 8400 voting tabulation districts (VTDs), which are consistent across all of the years.

Normal votes were calculated as the average share of the vote for the Democratic candidate across all contested elections in each VTD for each year or specified time period.


Shapefile: Texas_VTD
 - geography for each VTD
 - NV_D: Total Normal Votes for the Democratic Candidate, using the average Demcoratic vote for 2002-2010 weighted by 2008 Presidential Vote
 - NV_R: Total Normal Votes for the Republican Candidate, using the average Republican vote for 2002-2010 weighted by 2008 Presidential Vote
 - Gov, Pres: Total votes for each party for each gubernatorial and presidential race from 2002-2010.
 - vap: 2010 Census Voting Age Population

Data Files:

In all of the files, the prefix "t_" in a variable name indicates that the value is a total number of votes or people.  The prefix "p_" indicates a percentage, and "p_nv_" indicate a percentage normal vote.

Texas_General_Election_Results_2002-2010.data - election results for every federal and state level office.

Texas_General_Election_Candidates_2002-2010 - list of all candidate for every federal and state level office.

Texas_Normal_Vote_2002-2010 - normal votes calculated for each year and for 2002-2010, 2010 census data, and turnout information.

