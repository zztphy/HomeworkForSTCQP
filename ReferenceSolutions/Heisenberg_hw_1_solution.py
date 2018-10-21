"""
Exact diagonalization
Spin-1/2 antiferromagnetic Heisenberg model on a ring

Methods:
(1) use kronecker product
(2) do not apply any symmetry 
(3) apply U(1) symmetry
(4) apply translational symmetry
(5) apply U(1) and translational symmetries

Copyright:
Shuo Yang, shuoyang@tsinghua.edu.cn
Oct 12, 2018, Tsinghua, Beijing, China
"""

import numpy as np
import numpy.linalg as LA
from scipy import sparse
import math
#----------------------------------------------------------------
"""
Get spin operators
Input:
	ss = 2 means spin-1/2
	ss = 3 means spin-1
	ss = 4 means spin-3/2
"""
def SpinOper(ss):
	spin = (ss-1)/2.0
	dz = np.zeros(ss)
	mp = np.zeros(ss-1)
	
	for i in range(ss):
		dz[i] = spin-i
	for i in range(ss-1):
		mp[i] = np.sqrt((2*spin-i)*(i+1))
	
	S0 = np.eye(ss)
	Sp = np.diag(mp,1)
	Sm = np.diag(mp,-1)
	Sx = 0.5*(Sp+Sm)
	Sy = -0.5j*(Sp-Sm)
	Sz = np.diag(dz)
	
	return S0,Sp,Sm,Sz,Sx,Sy

"""
Bit operations	
"""
def SetBit(i,n):
	return i|(1<<n)

def ClearBit(i,n):
	return i&~(1<<n)

def FlipBit(i,n):
	return i^(1<<n)

def ReadBit(i,n):
	return (i&(1<<n))>>n

def PopCntBit(i):
	return bin(i).count("1")

def PickBit(i,k,n):
	return (i&((2**n-1)<<k))>>k

def RotLBit(i,L,n):
	return (PickBit(i,0,L-n)<<n)+(i>>(L-n))

def RotRBit(i,L,n):
	return (PickBit(i,0,n)<<(L-n))+(i>>n)
#----------------------------------------------------------------
"""
Get the list of hopping bonds
Input:
	Ns = 4
Output:
	HopList = [[0, 1], [1, 2], [2, 3], [3, 0]]
"""
def GetHopList(Ns):
	HopList = []
	for i in range(Ns):
		HopList.append([i,np.mod(i+1,Ns)])
	return HopList

"""
Build one term in Hamiltonian using kronecker product
Input:
	TempO = [Sx,Sx,S0,S0]
Output:
	TempH = np.kron(np.kron(np.kron(Sx,Sx),S0),S0)
"""
def GetHkron(TempO):
	Ns = len(TempO)
	
	TempH = TempO[0]
	for i in range(1,Ns):
		TempH = np.kron(TempH,TempO[i])
	
	return TempH

"""
Get basis
	'P' means particle number conserved
	'Basis' denotes the configuration in each Sz subspace
Input:
	Ns = 4
Output:
	Basis = [[0], [1, 2, 4, 8], [3, 5, 6, 9, 10, 12], [7, 11, 13, 14], [15]]
"""
def GetBasisP(Ns):
	Basis = [None]*(Ns+1)
	for Nu in range(Ns+1):
		Basis[Nu] = []
	
	for i in range(2**Ns):
		Nu = PopCntBit(i)
		Basis[Nu].append(i)
	
	return Basis

"""
Get basis
	'K' means momentum conserved
	'PriC' means the principle configuration
	After translating 'PriR' steps, the principle configuration go back to its original configuration
	0th column of 'Check': whether a configuration has been checked
	1st column of 'Check': which principle configuration can generate the current configuration by translation
	2nd column of 'Check': start from the principle configuration, how many translations to get the current configuration
	'Basis' denotes the allowed principle configuration in each [k] subspace
	
Input:
	Ns = 4
Output:
	PriC = [0, 1, 3, 5, 7, 15]
	PriR = [1, 4, 4, 2, 4, 1]
	Check = 
	[[ 0  0  0]
	 [ 0  1  0]
	 [ 1  1  1]
	 [ 0  3  0]
	 [ 1  1  2]
	 [ 0  5  0]
	 [ 1  3  1]
	 [ 0  7  0]
	 [ 1  1  3]
	 [ 1  3  3]
	 [ 1  5  1]
	 [ 1  7  3]
	 [ 1  3  2]
	 [ 1  7  2]
	 [ 1  7  1]
	 [ 0 15  0]]
	Basis = [[0, 1, 3, 5, 7, 15], [1, 3, 7], [1, 3, 5, 7], [1, 3, 7]]
"""
def GetBasisK(Ns):
	Nl = 2**Ns
	PriC = []
	PriR = []
	
	Check = np.zeros((Nl,3),dtype=int)
	for i in range(Nl):
		if Check[i,0] != 1:
			PriC.append(i)
			Check[i,1] = PriC[-1]
			Check[i,2] = 0
			for j in range(1,Ns):
				i1 = RotLBit(i,Ns,j)
				if i1 == i:
					R = j-1
					break
				Check[i1,0] = 1
				Check[i1,1] = PriC[-1]
				Check[i1,2] = j
				R = j
			PriR.append(R+1)
	
	Basis = [None]*Ns
	for k in range(Ns):
		Basis[k] = []
	
	for j in range(len(PriC)):
		R = PriR[j]
		for k in range(Ns):
			if np.mod(R*k,Ns) == 0:
				Basis[k].append(PriC[j])	
	
	return PriC,PriR,Check,Basis

"""
Get basis
	'P' means particle number conserved
	'K' means momentum conserved
	'PriC' means the principle configuration in each Sz subspace
	After translating 'PriR' steps, the principle configuration go back to its original configuration
	0th column of 'Check': whether a configuration has been checked
	1st column of 'Check': which principle configuration can generate the current configuration by translation
	2nd column of 'Check': start from the principle configuration, how many translations to get the current configuration
	'Basis' denotes the allowed principle configuration in each [Sz,k] subspace
	
Input:
	Ns = 4
Output:
	PriC = [[0], [1], [3, 5], [7], [15]]
	PriR = [[1], [4], [4, 2], [4], [1]]
	Check = 
	[[ 0  0  0]
	 [ 0  1  0]
	 [ 1  1  1]
	 [ 0  3  0]
	 [ 1  1  2]
	 [ 0  5  0]
	 [ 1  3  1]
	 [ 0  7  0]
	 [ 1  1  3]
	 [ 1  3  3]
	 [ 1  5  1]
	 [ 1  7  3]
	 [ 1  3  2]
	 [ 1  7  2]
	 [ 1  7  1]
	 [ 0 15  0]]
	Basis = 
	[[[0] [] [] []]
	 [[1] [1] [1] [1]]
	 [[3, 5] [3] [3, 5] [3]]
	 [[7] [7] [7] [7]]
	 [[15] [] [] []]]
"""
def GetBasisPK(Ns):
	Nl = 2**Ns
	PriC = [None]*(Ns+1)
	PriR = [None]*(Ns+1)
	for Nu in range(Ns+1):
		PriC[Nu] = []
		PriR[Nu] = []
	
	Check = np.zeros((Nl,3),dtype=int)
	for i in range(Nl):
		if Check[i,0] != 1:
			Nu = PopCntBit(i)
			PriC[Nu].append(i)
			Check[i,1] = PriC[Nu][-1]
			Check[i,2] = 0
			for j in range(1,Ns):
				i1 = RotLBit(i,Ns,j)
				if i1 == i:
					R = j-1
					break
				Check[i1,0] = 1
				Check[i1,1] = PriC[Nu][-1]
				Check[i1,2] = j
				R = j
			PriR[Nu].append(R+1)
	
	Basis = [None]*(Ns+1)*Ns
	Basis = np.reshape(Basis,[Ns+1,Ns])
	for Nu in range(Ns+1):
		for k in range(Ns):
			Basis[Nu,k] = []
	
	for Nu in range(Ns+1):
		for j in range(len(PriC[Nu])):
			R = PriR[Nu][j]
			for k in range(Ns):
				if np.mod(R*k,Ns) == 0:
					Basis[Nu,k].append(PriC[Nu][j])	
	
	return PriC,PriR,Check,Basis
#----------------------------------------------------------------
if __name__ == "__main__":
	Ns = 4
	Nl = 2**Ns
	
	HopList = GetHopList(Ns)
	print('HopList',HopList)
#-----------------------------------------------------
	"""
	(1) use kronecker product
	"""
	
	S0,Sp,Sm,Sz,Sx,Sy = SpinOper(2)
	
	Hamr = np.zeros((Nl,Nl))
	for ih in range(len(HopList)):
		Pos0 = HopList[ih][0]
		Pos1 = HopList[ih][1]
		
		TempO = [S0]*Ns
		TempO[Pos0] = Sp
		TempO[Pos1] = Sm/2.0
		TempH = GetHkron(TempO)
		Hamr += TempH
		
		TempO = [S0]*Ns
		TempO[Pos0] = Sm
		TempO[Pos1] = Sp/2.0
		TempH = GetHkron(TempO)
		Hamr += TempH
		
		TempO = [S0]*Ns
		TempO[Pos0] = Sz
		TempO[Pos1] = Sz
		TempH = GetHkron(TempO)
		Hamr += TempH
		
	# print Hamr
	# print np.diag(Hamr)
	S,V = LA.eigh(Hamr)
	print('(1) use kronecker product')
	print(S)
	
	Spm1 = np.sort(S)
	# print 'Spm1',Spm1
#-----------------------------------------------------
	"""
	(2) do not apply any symmetry
	"""
	
	HI = []
	HJ = []
	HV = []
	
	for i0 in range(Nl):
		for ih in range(len(HopList)):
			Pos0 = HopList[ih][0]
			Pos1 = HopList[ih][1]
			
			if ReadBit(i0,Pos0) != ReadBit(i0,Pos1):
				i1 = FlipBit(i0,Pos0)
				i1 = FlipBit(i1,Pos1)
				HI.append(i1)
				HJ.append(i0)
				HV.append(0.5)
				# print i0,np.binary_repr(i0,Ns),Pos0,Pos1,np.binary_repr(i1,Ns),i1,HV[-1]
			
			HI.append(i0)
			HJ.append(i0)
			HV.append((ReadBit(i0,Pos0)-0.5)*(ReadBit(i0,Pos1)-0.5))
	
	Hamr = sparse.coo_matrix((HV,(HI,HJ)),shape=(Nl,Nl)).tocsc()
	# print Hamr.todense()
	
	S,V = LA.eigh(Hamr.todense())
	print('(2) do not apply any symmetry')
	print(S)
	
	Spm2 = np.sort(S)
	# print 'Spm2',Spm2
#-----------------------------------------------------	
	"""
	(3) apply U(1) symmetry
	"""
	
	Basis = GetBasisP(Ns)
	print('Basis',Basis)
	print('(3) apply U(1) symmetry')
	Spm3 = []
	
	for Nu in range(Ns+1):
		HI = []
		HJ = []
		HV = []
		Nsz = len(Basis[Nu])
		
		for j0 in range(Nsz):
			i0 = Basis[Nu][j0]
			
			for ih in range(len(HopList)):
				Pos0 = HopList[ih][0]
				Pos1 = HopList[ih][1]
				
				if ReadBit(i0,Pos0) != ReadBit(i0,Pos1):
					i1 = FlipBit(i0,Pos0)
					i1 = FlipBit(i1,Pos1)
					j1 = Basis[Nu].index(i1)
					HI.append(j1)
					HJ.append(j0)
					HV.append(0.5)
					# print j0,i0,np.binary_repr(i0,Ns),Pos0,Pos1,np.binary_repr(i1,Ns),i1,j1,HV[-1]
				
				HI.append(j0)
				HJ.append(j0)
				HV.append((ReadBit(i0,Pos0)-0.5)*(ReadBit(i0,Pos1)-0.5))
		
		Hamr = sparse.coo_matrix((HV,(HI,HJ)),shape=(Nsz,Nsz)).tocsc()
		# print Hamr.todense()
		
		S,V = LA.eigh(Hamr.todense())
		print('Nu=',Nu,S)
		Spm3 = np.append(Spm3,S)
	
	Spm3 = np.sort(Spm3)
	# print 'Spm3',Spm3
#-----------------------------------------------------
	"""
	(4) apply translational symmetry
	"""
	
	PriC,PriR,Check,Basis = GetBasisK(Ns)
	print('PriC',PriC)
	print('PriR',PriR)
	print('Check',Check)
	print('Basis',Basis)
	Spm4 = []
	
	for k in range(Ns):
		if len(Basis[k]) > 0:
			print('k=',k)
			HI = []
			HJ = []
			HV = []
			Nk = len(Basis[k])
			
			for j0 in range(Nk):
				i0 = Basis[k][j0]
				
				for ih in range(len(HopList)):
					Pos0 = HopList[ih][0]
					Pos1 = HopList[ih][1]
					
					if ReadBit(i0,Pos0) != ReadBit(i0,Pos1):
						i1 = FlipBit(i0,Pos0)
						i1 = FlipBit(i1,Pos1)
						i1PriC = Check[i1,1]
						i1PriL = Check[i1,2]
						i1PriR = PriR[PriC.index(i1PriC)]
						i0PriR = PriR[PriC.index(i0)]
						try:
							j1 = Basis[k].index(i1PriC)
							HI.append(j1)
							HJ.append(j0)
							HV.append(np.sqrt(float(i0PriR)/float(i1PriR))*0.5*np.exp(1j*2*math.pi*k/float(Ns)*i1PriL))
							# print j0,i0,np.binary_repr(i0,Ns),Pos0,Pos1,np.binary_repr(i1,Ns),i1,i1PriC,i1PriL,i1PriR,i0PriR,j1,HV[-1]
						except ValueError:
							pass
					
					HI.append(j0)
					HJ.append(j0)
					HV.append((ReadBit(i0,Pos0)-0.5)*(ReadBit(i0,Pos1)-0.5))
			
			Hamk = sparse.coo_matrix((HV,(HI,HJ)),shape=(Nk,Nk)).tocsc()
			# print Hamk.todense()
		
			S,V = LA.eigh(Hamk.todense())
			print(S)
			Spm4 = np.append(Spm4,S)
	
	Spm4 = np.sort(Spm4)
	# print 'Spm4',Spm4
#-----------------------------------------------------	
	"""
	(5) apply U(1) and translational symmetry
	"""
	
	PriC,PriR,Check,Basis = GetBasisPK(Ns)
	print('PriC',PriC)
	print('PriR',PriR)
	print('Check',Check)
	print('Basis',Basis)
	Spm5 = []
	
	for Nu in range(Ns+1):
		for k in range(Ns):
			if len(Basis[Nu,k]) > 0:
				print('Nu=',Nu,'k=',k)
				HI = []
				HJ = []
				HV = []
				Nszk = len(Basis[Nu,k])
				
				for j0 in range(Nszk):
					i0 = Basis[Nu,k][j0]
					
					for ih in range(len(HopList)):
						Pos0 = HopList[ih][0]
						Pos1 = HopList[ih][1]
						
						if ReadBit(i0,Pos0) != ReadBit(i0,Pos1):
							i1 = FlipBit(i0,Pos0)
							i1 = FlipBit(i1,Pos1)
							i1PriC = Check[i1,1]
							i1PriL = Check[i1,2]
							i1PriR = PriR[Nu][PriC[Nu].index(i1PriC)]
							i0PriR = PriR[Nu][PriC[Nu].index(i0)]
							try:
								j1 = Basis[Nu,k].index(i1PriC)
								HI.append(j1)
								HJ.append(j0)
								HV.append(np.sqrt(float(i0PriR)/float(i1PriR))*0.5*np.exp(1j*2*math.pi*k/float(Ns)*i1PriL))
								# print j0,i0,np.binary_repr(i0,Ns),Pos0,Pos1,np.binary_repr(i1,Ns),i1,i1PriC,i1PriL,i1PriR,i0PriR,j1,HV[-1]
							except ValueError:
								pass
						
						HI.append(j0)
						HJ.append(j0)
						HV.append((ReadBit(i0,Pos0)-0.5)*(ReadBit(i0,Pos1)-0.5))
				
				Hamk = sparse.coo_matrix((HV,(HI,HJ)),shape=(Nszk,Nszk)).tocsc()
				# print Hamk.todense()
		
				S,V = LA.eigh(Hamk.todense())
				print(S)
				Spm5 = np.append(Spm5,S)
	
	Spm5 = np.sort(Spm5)
	# print 'Spm5',Spm5
#----------------------------------------------------------------
	print('Check different methods')
	print(LA.norm(Spm2-Spm1),LA.norm(Spm3-Spm1),LA.norm(Spm4-Spm1),LA.norm(Spm5-Spm1))
	
	