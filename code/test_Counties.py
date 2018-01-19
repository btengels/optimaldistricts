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
        N = np.sum(tmp)
        tmp_Gamma = np.diag(tmp).dot(Gamma)
        s = np.sum(tmp_Gamma,axis=0)
        output += np.diag(tmp).dot(N*Gamma-s) #This uses a funny quirk, where subtracting a vector from a matrix will execute the operation to all the rows. May need some massaging to fix, but something like this should work.
    return output



#The following is pseudocode for solving:
#Given w fixed
#Find a transportation plan from v to w
#Which minimizes the standard transport distance plus the county distance

county_param = .1
step_size = .1
dist_mat = get_dist_mat()
num_it = 100
Gamma = get_OT(v,w,dist_Mat)
df = dataframe

for i in range(num_it):
    lin_county = linearized_County_Cost(df)
    cost_mat = county_param*lin_county + dist_mat
    delta_Gamma = get_OT(v,w,cost_mat)
    Gamma = Gamma + delta_Gamma*step_size

#TODO here: pull functions from make_maps to get this to work (i.e. make it not pseudocode). Then run it on PA.
#Note: Lloyd's algorithm actually works here without any modification. This is because the county cost depends on
#Gamma and not w, so we can do the gradient descent steps in w in exactly the same way as before (no change to code).

