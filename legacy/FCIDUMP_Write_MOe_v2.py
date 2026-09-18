#!/usr/bin/env python
import QCOpMat as qco
import QCMatEl as qcm
import numpy as np
import sys

#INPUT SECTION , format of Input : python FCIDUMP_Write.py matfile active_orbitals
fileinp = sys.argv[1]
filemat = f'{fileinp}.mat'
me = qcm.MatEl(file=filemat)
#Read number of basis functions (nmo), number of MOs (nact) and the number of frozen core (nfc). 
nmo = me.nbsuse
nao = me.nbasis
ncore = int(me.nfc)
nact = nmo - ncore - me.nfv
nelec = me.ne - me.nfc*2
nactocc = int(ncore + nelec/2)
print(f"there are {nact} active orbitals and {nelec} active electrons and a total of {nactocc} occupied orbitals and {nmo} total orbitals")

#Read the integrals, resize them according to the size of the active space
me.T = me.matlist["KINETIC ENERGY"].expand().copy()
me.T.resize((nao,nao))
me.V = me.matlist["CORE HAMILTONIAN ALPHA"].expand().copy()
me.V.resize((nao,nao))
#Read MO coefficients and transform 1-electron integrals to MO basis
me.CMO = me.matlist["ALPHA MO COEFFICIENTS"].array.copy()
me.CMO.resize(nmo,nao)
h1e = np.dot(me.CMO, np.dot(me.V, np.transpose(me.CMO)))
#Read 2-electron integrals in MO basis
h2e=me.matlist["AA MO 2E INTEGRALS"].expand().copy()
h2e.resize(nact,nact,nact,nact)
#Read nuclear repulsion energy, number of electrons and spin
enuc = me.scalar('ENUCREP')
ms2= me.multip-1
#Read the molecular orbital energies for the next part
me.MOE=me.matlist["ALPHA ORBITAL ENERGIES"].expand().copy()
me.MOEd = np.diag(me.MOE)

#Manipulations to Reduce the Hamiltonian Space to the Active Space
nact_2e = nactocc - ncore
iact = ncore + nact
h1e_act = me.MOEd[ncore:iact,ncore:iact]
print(f"the shape of Vmod is {h1e_act.shape},there are {nact_2e} active occupied orbitals")
h1e_act -= 2*np.einsum('acbb->ac', h2e[:nact, :nact, :nact_2e, :nact_2e])
h1e_act += np.einsum('abbc->ac', h2e[:nact, :nact_2e, :nact_2e, :nact])

#NuclearRepulsion Term
enuc +=  np.trace(h1e[:nactocc, :nactocc])
enuc += np.trace(me.MOEd[:ncore,:ncore])
enuc -=  np.trace(h1e_act[:nact_2e, :nact_2e])
print(f"The nuclear repulsion is {enuc}")

# Write FCIDUMP file
f = open(f'/work/lgutsev/gauopen/{fileinp}.dat','w+')
f.write(f'&FCI NORB = {nact}, NELEC={nelec} , MS2={ms2}, \n')
f.write(' ORBSYM=')
for i in range(nact):
  f.write('1,')
f.write('\n')
f.write(' ISYM=1, \n')
f.write('&END \n')
# Write 2-electron integrals
for i in range(nact):
 for j in range(0,i+1):
   for k in range(0,i+1):
    if i == k:
      last = j+1
    else:
      last = k + 1
    for l in range(0,last):
      value = h2e[i, j, k, l]
      f.write(f'{value:24.16E} {i+1:4d} {j+1:4d} {k+1:4d} {l+1:4d} \n')
# Write 1-electron integrals
for i in range(nact):
 for j in range(0,i+1):
  f.write(f'{h1e_act[i,j]:24.16E}')
  f.write(f' {i+1:4d} {j+1:4d} 0 0 \n')
# Write nuclear repulsion
f.write(f'{enuc:24.16E} 0 0 0 0\n')
f.close()







