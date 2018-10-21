#exact diagonalize transvers field Ising model with periodical boundary condition 
import numpy as np
import matplotlib.pyplot as plt

#initial parameters
N = 4  #num of sites
J = 1
g = 0.5
###########################################################
#binary operations
def PopCntBit(i): #count the number of 1
    return bin(i).count('1')

def PickBit(i, k, n): #pick up n bits from kth bit
    return (i & ((2**n - 1) << k)) >> k

def RotLBit(i, L, n): #circular bit shift left
    return (PickBit(i, 0, L-n) << n) + (i >> (L-n))

def RotRBit(i, L, n): #circular bit shift right
    return (PickBit(i, 0, n) << (L-n)) + (i >> n)

def ReadBit(i, n):  #read nth bit
    return (i & (1 << n)) >> n

def FlipBit(i, n): #flip nth bit
    return i^(1 << n)

##########################################

def CalPriC(N):
    PriC = list()
    for _i in range(2**N):
        PriC.append(_i)
    for _j in range(2**N):
        if _j >= len(PriC):
            break
        else:
            _PriC = PriC[_j]
            for _k in range(N):
                if (RotLBit(_PriC, N, _k)!=_PriC) & (RotLBit(_PriC, N, _k) in PriC):
                    PriC.remove(RotLBit(_PriC, N, _k))
    return PriC

def CalPriR(PriC):
    PriR = list()
    for _i in range(len(PriC)):
        _PriC = PriC[_i]
        count = 1
        while 1:
            if RotLBit(_PriC, N, count) == _PriC:
                break
            count += 1
        PriR.append(count)
    return PriR

def CalBasis(N, PriC, PriR):
    Basis = list()
    for _i in range(N):
        Basis.append(list())
        for _j in range(len(PriC)):
            if (_i*PriR[_j]) % N == 0:
                Basis[_i].append(PriC[_j])
    return Basis

def CalCheck(N, PriC):
    Check = list()
    for i in range(2**N):
        Check.append(list())
        if i in PriC:  #1 for yes
            Check[i].append(1)
        else:
            Check[i].append(0)

        count = 0
        while 1:
            if RotRBit(i, N, count) in PriC:
                break
            count += 1
        Check[i].append(RotRBit(i, N, count))
        Check[i].append(count)
    return Check

def CalHami(N):
    PriC = CalPriC(N)
    PriR = CalPriR(PriC)
    Basis = CalBasis(N, PriC, PriR)
    Check = CalCheck(N, PriC)

    Hamitonian = list()
    for m in range(N): #loop for every subspace k(or m)
        Hamitonian.append(np.zeros([len(Basis[m]), len(Basis[m])], dtype = 'complex'))
        if len(Basis[m]) == 0:
            continue
        
        for j in range(N): #loop for every H_j
#calculate off diag terms
            for col_index in range(len(Basis[m])):
                a = Basis[m][col_index]
                b_prime = FlipBit(a, j)
                b = Check[b_prime][1]
                l_j = Check[b_prime][2]

                _index = PriC.index(a)
                R_a = PriR[_index]
                _index = PriC.index(b)
                R_b = PriR[_index]

                if b not in Basis[m]:
                    continue
                temp = -J * g * np.sqrt(R_a/R_b) * complex(np.cos(2*np.pi*m*l_j/N), -np.sin(2*np.pi*m*l_j/N))
                row_index = Basis[m].index(b)
                Hamitonian[m][row_index, col_index] += temp

#calculate diag terms
            for col_index in range(len(Basis[m])):
                a = Basis[m][col_index]
                sigma_j_z = ReadBit(a, j)
                if j+1 <= N-1:
                    sigma_jp1_z = ReadBit(a, j+1)
                else:
                    jj = j+1
                    while 1:
                        if jj <= N-1:
                            break
                        else:
                            jj = jj - N
                    sigma_jp1_z = ReadBit(a, j+1)
                row_index = col_index
                Hamitonian[m][row_index, col_index] += -J * sigma_j_z * sigma_jp1_z
    return Hamitonian


####main####
H = CalHami(N)
print('Check Hamitonian, please input k(m)...:')
k_temp = input()
print('H in this subspace is:\n')
print(H[int(k_temp)])
EigenValues = list()
for _ in range(N):
    EigenValues.append(list())
for m in range(N):
    if H[m].shape[0] != 0:
        eigenvalue, eigenvector = np.linalg.eig(H[m])
        EigenValues[m] = EigenValues[m] + list(eigenvalue)
for _ in range(N):
    EigenValues[_] = np.real(EigenValues[_])
    EigenValues[_].sort()

print('#####Eigenvalues are:#####\n')
for _ in range(N):
    print('m = %i: ' %_)
    print(EigenValues[_])
    print('\n')         

exact_ground = list()
epsilo_0 = 0
for m in range(N):
    epsilo_0 += np.sqrt(1+g**2-2*g*np.cos(2*np.pi*m/N))
epsilo_0 = epsilo_0 * J
for m in range(N):
    exact_ground.append(2*J*np.sqrt(1+g**2-2*g*np.cos(2*np.pi*m/N))-epsilo_0)
cal_ground = list()
for m in range(N):
    cal_ground.append(EigenValues[m][0])
print('Exact groud state eigenvalues and calculated groud state eigenvalues are shown below:\n')
plt.plot(exact_ground)
plt.plot(cal_ground)