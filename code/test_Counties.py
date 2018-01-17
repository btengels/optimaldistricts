# test_Counties.py
# Written by Ryan Murray 1/16/18

# The goal of this script is to build a function necessary to computing optimal voting districts which try to take into account geographic communities of interest. The companion paper proposed the following cost:
#   \sum_{g(i) = g(j)} \|\Gamma_i-\Gamma_j\|^2     (**)
#
# where here g is a function mapping districts to some geographic community and Gamma_i and Gamma_j are the ith and jth rows of the transportation plan Gamma.

#Inputs: Gamma - Current transportation plan
#        df    - Data frame.

#Output: Linearization of the cost (**) about the current transportation plan.

def linearized_County_Cost(Gamma,df):
    county_names = np.unique(df['COUNTY_NAM'])
    output = np.zeros(Gamma.shape)
    for county in county_names:
        tmp = (df['COUNTY_NAM'] == county)
        tmp_Gamma = np.diag(tmp).dot(Gamma)
        avg = np.sum(tmp_Gamma,axis=0)
        output += np.diag(tmp).dot(Gamma-avg) #This uses a funny quirk, where subtracting a vector from a matrix will execute the operation to all the rows. May need some massaging to fix, but something like this should work.
    return output
