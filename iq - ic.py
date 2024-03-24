import numpy as np
from pylab import plot, xlim, ylim
import math

#-----------------------basis--------
bra0 = np.zeros([1,2])
bra0[0][0]=1

ket0 = np.zeros([2,1])
ket0[0][0]= 1

bra1 = np.zeros([1,2])
bra1[0][1]=1

ket1 = np.zeros([2,1])
ket1[1][0]=1

ket00 = np.kron(ket0, ket0)
bra00 = np.kron(bra0, bra0)
ket11 = np.kron(ket1, ket1)
bra11 = np.kron(bra1, bra1)

#--------------identity--------

ii = np.identity(4)

#-----------------bell state--------

bell1 = (1/(np.sqrt(2))) * (ket00 + ket11)
bell2 = (1/(np.sqrt(2))) * (bra00 + bra11)

bell = np.kron(bell1, bell2) #this is non-diagonalized

#-----------------werner state------
zz=[]
A=[]
QA = []
deltai=[]
p0 = 0.25
p1 = 1 - p0
deltabar = 0

#--------------identity--------

ii = np.identity(4)
i1 = np.identity(2)
#------------------------------

def entropy( s):
    ent = 0
    ev , es = np.linalg.eig(s)
    for i in ev:
        if i>0:
            ent += -i.real*math.log2(i.real)
    return ent

#-------------------------partial trace function----------------

def ptrace(y):
    x = np.zeros([2,2])
    x[0][0] = y[0][0] + y[1][1] 
    x[0][1] = y[0][2] + y[1][3]
    x[1][0] = y[2][0] + y[3][1]
    x[1][1] = y[2][2] + y[3][3]
    return x

#--------------------------povm function--------------------

def povm( pvm1 , state):
    mes =  np.matmul(np.matmul(pvm1, state),np.transpose(np.conjugate(pvm1)))
    
    rhoc = np.zeros((2,2))
    rhoc = ptrace(mes)
    
    rhoc = rhoc/np.trace(np.matmul(np.matmul(np.transpose(np.conjugate(pvm1)),pvm1), state))
    return rhoc


def pmeas(x,y):
    pi = np.trace(np.dot(np.dot(np.transpose(np.conjugate(x)),x), y))
    return pi



for i in range(1000):
    z=i/1000
    w = (1-z)*ii/4 + z*bell
    
#---------------sub state-----------    

    rhob = ptrace(w)
    
#---------------measurement operator------
#it has to acton diagonalized matrices
    povm1 = np.kron(np.identity(2) , np.kron(ket0, bra0))
    povm2 = np.kron(np.identity(2) , np.kron(ket1, bra1))
    
#----------------conditional state------------

    rhoc21 = povm(povm1, w)
    rhoc22 = povm(povm2, w)

    
#--------------------average state----------

    rhobar = p0 * w + p1*ii/4
    
#-------------------measurement on average state----

    rhobarc1 = p0*rhoc21 + p1*i1/2
    rhobarc2 = p0*rhoc22 + p1*i1/2

#-----------average substate-----------

    rhobarb = ptrace(rhobar)

#------------------entropies

    srhow = 0
    srhob = 0
    srhoc = 0 
    srhow += entropy(w)
    srhob += entropy(rhob)
    srhoc += pmeas(povm1, w)*entropy( rhoc21) + pmeas(povm2, w)*entropy( rhoc22)

    
    
    srhobarc = 0
    srhobarb = 0
    srhobar = 0
    
    srhobar += entropy(rhobar)
    srhobarb += entropy(rhobarb)
    srhobarc +=  pmeas(povm1, w)*entropy( rhobarc1) + pmeas(povm2, w)*entropy( rhobarc2)
            
    Iq =  srhobar - p0*srhow - p1*(srhob + 1)
    Ic = srhobarc - p0*srhoc - p1*srhob
    deltaI = Iq - Ic
    deltai.append(deltaI)
    A.append(Ic)
    QA.append(Iq)
    zz.append(z)
print(srhobar)   
#plot(zz , A, color='black' )
#plot(zz , QA, color='y')
plot(zz , deltai,color='b')

print(w)
xlim(0, 0.2)
ylim(0, 0.04)
