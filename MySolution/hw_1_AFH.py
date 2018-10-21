#exact diagonalize spin 1/2 AF heisenberg model with periodical boundary condition 
import numpy as np
import matplotlib.pyplot as plt

#number of sites
N = 4

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

def Flip(i, n, s, L):
    while n > L-1:
        n = n - L
    if s == 'up':
        if ReadBit(i, n) == 1:
            return -1
        else:
            return FlipBit(i, n)
    if s == 'down':
        if ReadBit(i, n) == 0:
            return -1
        else:
            return FlipBit(i, n)

#find the index for elements in a 2-D array
def FindIndex(value, lis):
    for i in range(len(lis)):
        for j in range(len(lis[i])):
            if value == lis[i][j]:
                return i,j
    return -1,-1

#calculate PriC[Nu][j]
def PrinC(N):
    PriC = list()
    for i in range(N+1):
        PriC.append(list())
    for i in range(2**N):
        PriC[PopCntBit(i)].append(i)
    for i in range(N+1):
        len_temp = len(PriC[i])
        for j in range(len_temp):
            if j > len(PriC[i])-1:
                break
            else:
                temp = PriC[i][j]
                for k in range(N):
                    if(RotLBit(temp, N, k) in PriC[i]):
                        if RotLBit(temp, N, k) != temp:
                            PriC[i].remove(RotLBit(temp, N, k))
    return PriC

#calculate PriR[Nu][j]
def PrinR(PriC):
    PriR = list()
    for i in range(len(PriC)):
        PriR.append(list())
        for j in range(len(PriC[i])):
            PriR[i].append(list())
    for i in range(len(PriC)):
        for j in range(len(PriC[i])):
            temp = PriC[i][j]
            count = 1
            while 1:
                if RotLBit(temp, N, count) == temp:
                    break
                else:
                    count+=1
            PriR[i][j] = count
    return PriR

#calculate Basis
def CalBasis(N, PriC, PriR):
    Basis = list()
    for i in range(N+1):
        Basis.append(list())
        for j in range(N):
            Basis[i].append(list())
            len_temp = len(PriR[i])
            for k in range(len_temp):
                if (j*PriR[i][k]) % N == 0:
                    Basis[i][j].append(PriC[i][k])
    return Basis

#calculate Check matrix
def CalCheck(N, PriC):
    Check = list()
    for i in range(2**N):
        Check.append(list())
        if i in PriC[PopCntBit(i)]:  #1 for yes
            Check[i].append(1)
        else:
            Check[i].append(0)

        count_temp = 0
        Nu_temp = PopCntBit(i)
        while 1:
            if RotRBit(i, N, count_temp) in PriC[Nu_temp]:
                break
            count_temp+=1
        Check[i].append(RotRBit(i, N, count_temp))
        Check[i].append(count_temp)
    return Check
            
#Calculate H in every subspace(Nu, k)
def CalHami(N):
    PriC = PrinC(N)
    PriR = PrinR(PriC)
    Basis = CalBasis(N, PriC, PriR)
    Check = CalCheck(N, PriC)

#build empty Hamitonian
    Hamitonian = list()
    for i in range(N+1):
        Hamitonian.append(list())
        for j in range(N):
            Hamitonian[i].append(np.zeros( [len(Basis[i][j]), len(Basis[i][j])], dtype='complex'))

    for S_z in range(N+1):
        for m in range(N):
            if len(Basis[S_z][m]) == 0:
                continue
            for j in range(N):

#calculate off diagonal terms H_j^off
                for col_index in range(len(Basis[S_z][m])):
                    a_k = Basis[S_z][m][col_index]
                    if (Flip(a_k, j, 'up', N)!=(-1)) & (Flip(a_k, j+1, 'down', N)!=(-1)):
                        b_j_prime = Flip(a_k, j, 'up', N)
                        b_j_prime = Flip(b_j_prime, j+1, 'down', N)
                        b_j = Check[b_j_prime][1]
                        l_j = Check[b_j_prime][2]

                        _1,_2 = FindIndex(a_k, PriC)
                        PriR_a = PriR[_1][_2]
                        _1,_2 = FindIndex(b_j, PriC)
                        PriR_b_j = PriR[_1][_2]
                        h_j = 0
                        for _ in range(N):
                            if RotLBit(b_j_prime, N, _) in Basis[S_z][m]:
                                h_j = 0.5
                                break
                        if not (b_j in Basis[S_z][m]):
                            continue
                        else:
                            temp = np.sqrt(PriR_a/PriR_b_j)*h_j*complex(np.cos(2*np.pi*m*l_j/N), -np.sin(2*np.pi*m*l_j/N))
                            row_index = Basis[S_z][m].index(b_j)
                            Hamitonian[S_z][m][row_index, col_index] += temp
                    elif (Flip(a_k, j, 'down', N)!=(-1)) & (Flip(a_k, j+1, 'up', N)!=(-1)):
                        b_j_prime = Flip(a_k, j, 'down', N)
                        b_j_prime = Flip(b_j_prime, j+1, 'up', N)
                        b_j = Check[b_j_prime][1]
                        l_j = Check[b_j_prime][2]

                        _1,_2 = FindIndex(a_k, PriC)
                        PriR_a = PriR[_1][_2]
                        _1,_2 = FindIndex(b_j, PriC)
                        PriR_b_j = PriR[_1][_2]
                        h_j = 0
                        for _ in range(N):
                            if RotLBit(b_j_prime, N, _) in Basis[S_z][m]:
                                h_j = 0.5
                                break
                        if not (b_j in Basis[S_z][m]):
                            continue
                        else:
                            temp = np.sqrt(PriR_a/PriR_b_j)*h_j*complex(np.cos(2*np.pi*m*l_j/N), -np.sin(2*np.pi*m*l_j/N))
                            row_index = Basis[S_z][m].index(b_j)
                            Hamitonian[S_z][m][row_index, col_index] += temp
                    else:
                        continue


#calculate diagonal terms H_j^diag
                for col_index in range(len(Basis[S_z][m])):
                    a_k = Basis[S_z][m][col_index]
                    S_j_z = ReadBit(a_k, j)-0.5
                    if j+1 <= N-1:
                        S_jp1_z = ReadBit(a_k, j+1) - 0.5
                    else:
                        jj = j+1
                        while 1:
                            if jj <= N-1:
                                break
                            else:
                                jj = jj - N
                        S_jp1_z = ReadBit(a_k, jj) - 0.5
                    row_index = col_index
                    Hamitonian[S_z][m][row_index, col_index] += S_j_z * S_jp1_z

    return Hamitonian 




###main
H = CalHami(N)
print('Check Hamitonian, please input S_z and k(m)...:')
S_z_temp = input()
k_temp = input()
print('H in this subspace is:\n')
print(H[int(S_z_temp)][int(k_temp)])
EigenValues = list()
for _ in range(N):
    EigenValues.append(list())
for m in range(N):
    for S_z in range(N+1):
        if H[S_z][m].shape[0] != 0:
            eigenvalue, eigenvector = np.linalg.eig(H[S_z][m])
            EigenValues[m] = EigenValues[m] + list(eigenvalue)
for _ in range(N):
    EigenValues[_] = np.real(EigenValues[_])
    EigenValues[_].sort()

print('#####Eigenvalues are:#####\n')
for _ in range(N):
    print('m = %i: ' %_)
    print(EigenValues[_])
    print('\n')         