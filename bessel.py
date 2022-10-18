import numpy as np
from scipy import special
from scipy import optimize
import matplotlib.pyplot as plt

"""
rho = np.linspace(-20,20,200)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,6))

j0 = sp.jv(0, rho)
j1 = sp.jv(1, rho)
j2 = sp.jv(2, rho)
j3 = sp.jv(3, rho)

y0 = sp.yn(0, rho)
y1 = sp.yn(1, rho)
y2 = sp.yn(2, rho)
y3 = sp.yn(3, rho)

i0 = sp.iv(0, rho)
i1 = sp.iv(1, rho)
i2 = sp.iv(2, rho)
i3 = sp.iv(3, rho)

k0 = sp.kn(0, rho)
k1 = sp.kn(1, rho)
k2 = sp.kn(2, rho)
k3 = sp.kn(3, rho)

axes[0,0].plot(rho, j0, label='$J_0(x)$')
axes[0,0].plot(rho, j1, label='$J_1(x)$')
axes[0,0].plot(rho, j2, label='$J_2(x)$')
axes[0,0].plot(rho, j3, label='$J_3(x)$')
axes[0,0].legend(loc='best')
axes[0,0].grid(True)
axes[0,0].set_title('Bessel function of a first kind')

axes[0,1].plot(rho, y0, label='$Y_0(x)$')
axes[0,1].plot(rho, y1, label='$Y_1(x)$')
axes[0,1].plot(rho, y2, label='$Y_2(x)$')
axes[0,1].plot(rho, y3, label='$Y_3(x)$')
axes[0,1].legend(loc='best')
axes[0,1].grid(True)
axes[0,1].set_title('Bessel function of a second kind')
axes[0,1].set_xlim(0,20)
axes[0,1].set_ylim(-3, .7)

axes[1,0].plot(rho, i0, label='$I_0(x)$')
axes[1,0].plot(rho, i1, label='$I_1(x)$')
axes[1,0].plot(rho, i2, label='$I_2(x)$')
axes[1,0].plot(rho, i3, label='$I_3(x)$')
axes[1,0].legend(loc='best')
axes[1,0].grid(True)
axes[1,0].set_title('Modified Bessel function of a first kind')
axes[1,0].set_xlim(0,4)
axes[1,0].set_ylim(0,4)

axes[1,1].plot(rho, k0, label='$K_0(x)$')
axes[1,1].plot(rho, k1, label='$K_1(x)$')
axes[1,1].plot(rho, k2, label='$K_2(x)$')
axes[1,1].plot(rho, k3, label='$K_3(x)$')
axes[1,1].legend(loc='best')
axes[1,1].grid(True)
axes[1,1].set_title('Modified Bessel function of a second kind')
axes[1,1].set_xlim(0,4)
axes[1,1].set_ylim(0,4)

fig.tight_layout()
fig.savefig('./bessels.png', dpi=300, bbox_inches='tight')
#fig.savefig('./bessels.png', dpi=300)
"""

"""
a = 8 # um.
n1 = 1.46
n2 = 1.44
k1 = 2*np.pi * n1 / 0.94 # 940 nm, in unit of um.
k2 = 2*np.pi * n2 / 0.94 # 940 nm, in unit of um.
upperlim = np.sqrt(k1**2-k2**2)
#upperlim = 2
print(upperlim)
k_rho = np.linspace(0,upperlim,100)
f1 = sp.jv(1, k_rho*a) / sp.jv(0, k_rho*a)
f2 = (sp.kv(-1,k_rho*a)+sp.kv(1, k_rho*a)) / 2 / sp.kv(0, k_rho*a)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6,3))
axes.plot(k_rho, f1, label='LHS')
axes.plot(k_rho, f2, label='RHS')
axes.set_ylim(-.5, upperlim)
axes.set_xlabel(r'$k_{\rho}$')
axes.legend(loc='best')
fig.tight_layout()
fig.savefig('./LP0mode.png', dpi=300, bbox_inches='tight')
#fig.savefig('./bessels.png', dpi=300)
민주짱
"""

class stepidxoptfiber:

    def __init__(self, a, n1, n2, wavelength):
        """Setup an optical fiber with given parameters.

        Parameters
        ----------
        a: float
            A radius of the core in unit of um.

        n1: float
            An index of the core.

        n2: float
            An index of the cladding.

        n: int
            'n' of the Bessel function of 1st kind
            or the modified Bessel function of 2nd kind.

        wavelength: float
            A wavelength in air in unit of um.

        Returns
        -------
        None
        """

        self.cr = a # the radius of the core.
        self.n1 = n1 # the index of the core.
        self.n2 = n2 # the index of the cladding.
        self.wvl = wavelength # the wavelength of the input wave.

        self.k0 = 2*np.pi / self.wvl 
        self.k1 = 2*np.pi * n1 / self.wvl
        self.k2 = 2*np.pi * n2 / self.wvl

        self.hs = np.linspace(self.k2, self.k1, 600)

    def find_RHS(self, n, x, y, k1, k2):

        eta1 = (special.jvp(n,x)/x/special.jv(n,x))
        eta2 = (special.kvp(n,y)/y/special.kv(n,y))

        val = (eta1*(k1**2) + eta2*(k2**2)) * (eta1 + eta2) 

        return val

    def find_LHS(self, n, h, x, y):

        deno = ((1/x**2 + 1/y**2)**2)
        val = (n*h)**2 * deno

        return val

    def find_h(self, h, n):
        """Numerically find h in a step-index optical fiber.

        Parameters
        ----------
        n: int
            'n' of the Bessel function of 1st kind
            or the modified Bessel function of 2nd kind.

        Returns
        -------
        None
        """

        self.u = np.sqrt(self.k1**2 - h**2)
        self.w = np.sqrt(h**2 - self.k2**2)
        self.x = self.u * self.cr 
        self.y = self.w * self.cr

        RHS = self.find_RHS(n, self.x, self.y, self.k1, self.k2)
        LHS = self.find_LHS(n, h, self.x, self.y)
        val = RHS - LHS

        return val

    def find_roots(self,):

        x0, r = optimize.brentq(self.find_h, self.k2, self.k1, args=(self.n))

        return x0, r

    def plot(self, x, y, eta1, eta2):

        self.LHS = self.find_LHS()

        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9,3))
        axes[0].plot(self.hs, self.LHS, label='LHS')
        axes[0].plot(self.hs, self.RHS, label='RHS')
        axes[1].plot(self.hs, self.val, label='RHS-LHS')

        if n == 0: axes[0].set_ylim(-.1, 0.2)
        else: axes[0].set_ylim(-.2, 5)

        fig.suptitle(f'n={n}')
        axes[1].set_ylim(-.3, 0.3)
        axes[1].hlines(0, self.k2, self.k1, 'gray', '--')
        #axes.set_ylim(-.5, np.max(LHS)*1.2)
        #axes.set_xlabel(r'$k_{\rho}$')
        axes[0].set_xlabel(r'$h$')
        axes[1].set_xlabel(r'$h$')
        axes[0].legend(loc='best')
        axes[1].legend(loc='best')
        fig.tight_layout()
        fig.savefig(f'.//finding_h_n{n}.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    a = 8 # um.
    n1 = 1.46
    n2 = 1.44
    wvl = 0.94 # um.

    fiber = stepidxoptfiber(a, n1, n2, wvl)

    for n in range(10): 
        func = fiber.find_h(n)
        x0, r = fiber.find_roots()