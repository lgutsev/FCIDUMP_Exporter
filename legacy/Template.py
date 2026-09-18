from pyscf import gto, scf, mcscf

mol = gto.Mole()
mol.atom= '''
;
'''

mol.build(
verbose = 5,
output = '/work/lgutsev/DICE/Template/Template.out',
basis = {'N':'anoroostz','Ni':'anoroostz'},
spin = 2
)

# Create HF molecule
mf = scf.UHF( mol )
mf.conv_tol = 1e-8
mf.chkfile = '/work/lgutsev/DICE/Template/Template.chk'
#mf.kernel()
mf.scf()

from pyscf import cornell_shci as shci

# Number of orbital and electrons
norb = 21
nelec = 16

# Create SHCI molecule for just variational opt.
# Active spaces chosen to reflect valence active space.
mch = shci.SHCISCF( mf, norb, nelec )
mch.fcisolver.mpiprefix = 'mpirun -np 2'
mch.fcisolver.stochastic = True
mch.fcisolver.nPTiter = 0
mch.fcisolver.sweep_iter = [ 0, 3 ]
mch.fcisolver.DoRDM = True
mch.fcisolver.sweep_epsilon = [ 5e-3, 1e-3 ]
e_shci = mch.mc1step()[0]


