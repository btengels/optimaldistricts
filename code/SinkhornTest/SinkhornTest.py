import numpy as np
import sklearn.metrics as metrics

def computeTransportSinkhorn(distribS,distribT, M, reg,uin):
	# Sinkhorn algorithm from Cuturi 2013.
	'''
	M is a matrix of pairwise distances
	distribS is a distribution of initial weights
	distribT is a distribution of final weights
	reg is a regularization parameter
	uin is an initial guess for the Lagrange multiplier u
	'''
	# init data
	Nini = len(distribS)
	Nfin = len(distribT)

	numItermax = 2000
	cpt = 0

	# we assume that no distances are null except those of the diagonal of distances
	#u = np.ones(Nini)/Nini
	u = uin
	uprev=np.zeros(Nini)

	K = np.exp(-reg*M)
	Kp = np.dot(np.diag(1/distribS),K)
	transp = K
	cpt = 0
	err=1


	while (err>1e-8 and cpt<numItermax):
		if np.logical_or(np.any(np.dot(K.T,u)==0),np.isnan(np.sum(u))):
			# we have reached the machine precision
			# come back to previous solution and quit loop
			print 'Infinity'
			if cpt!=0:
				u = uprev
			break
		uprev = u
		v = np.divide(distribT,np.dot(K.T,u))
		u = 1./np.dot(Kp,v)
		if cpt%3==0:
			# Currently checking for stopping every 5th iteration, may want to change
			transp = np.dot(np.diag(u),np.dot(K,np.diag(v)))
			err = np.linalg.norm((np.sum(transp,axis=0)-distribT))**2
		cpt = cpt +1

#
	print cpt
	Mout = np.dot(np.diag(u),np.dot(K,np.diag(v)))
	temp = np.log(Mout)
	mask = np.isreal(np.log(Mout))
	temp[mask] = 0
	cost = np.sum(M*Mout) + 1.0/reg*np.sum( temp*Mout )
	return Mout,u,cost


numPts = 12
h = 2

s = np.linspace(0,np.pi,numPts)
I = np.array([s,h*np.sin(s)]).T
F = np.array([s,h*np.cos(s)]).T
I_wgt = np.ones(numPts)/numPts
F_wgt = I_wgt

reg = 200

#I = Iin
#F = Fin
Nini = len(I_wgt)
uinit = np.ones(Nini)/Nini

#Initial optimization step, mostly to initialize u
DistMat = metrics.pairwise.euclidean_distances( I, F )
Mout,u,cost = computeTransportSinkhorn(I_wgt, F_wgt, DistMat, reg,uinit)

print Mout
