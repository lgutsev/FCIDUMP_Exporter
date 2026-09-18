from pyscf import gto, scf, mcscf
from pyscf import cornell_shci as shci

# Number of orbital and electrons
norb = 82
nelec = 22

# Create SHCI molecule for just variational opt.
# Active spaces chosen to reflect valence active space.
mch = shci.SHCISCF( mf, norb, nelec )
mch.fcisolver.mpiprefix = 'mpirun -np 1'
mch.fcisolver.stochastic = True
mch.fcisolver.nPTiter = 0
mch.fcisolver.sweep_iter = [ 0, 3 ]
mch.fcisolver.DoRDM = True
mch.fcisolver.sweep_epsilon = [ 5e-3, 1e-3 ]
e_shci = mch.mc1step()[0]

