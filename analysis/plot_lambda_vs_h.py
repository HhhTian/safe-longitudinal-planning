"""
Plot effective multiplier lambda as function of safety margin h
for CBF, Quadratic, Log barrier, Exponential.

Run: python plot_lambda_vs_h.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.size'] = 12
rcParams['font.family'] = 'serif'
rcParams['figure.dpi'] = 150

# Parameters
G = 1.0
dt = 0.1
gamma = 5.0
w = 100.0
L_default = 0.5
f0_default = -5.0

h_range = np.linspace(-12, 8, 800)


def h_next(hk, u, f0=f0_default):
    return hk + (f0 + G * u) * dt


# 1. CBF
def cbf_multiplier(h_k, f0=f0_default):
    b = f0 + gamma * h_k
    return np.maximum(0, -b / G**2)


# 2. Quadratic: 0.5*u^2 + w*max(0, -h_{k+1})^2
def quadratic_multiplier(h_k_arr, f0=f0_default):
    out = np.zeros_like(h_k_arr)
    for i, hk in enumerate(h_k_arr):
        def cost(u):
            hk1 = h_next(hk, u, f0)
            return 0.5 * u**2 + w * max(0.0, -hk1)**2
        res = minimize_scalar(cost, bounds=(-10, 300), method='bounded')
        out[i] = res.x / G
    return out


# 3. Log barrier: 0.5*u^2 - mu*ln(h_{k+1})
def log_barrier_multiplier(h_k_arr, f0=f0_default, mu_bar=1.0):
    out = np.full_like(h_k_arr, np.nan)
    for i, hk in enumerate(h_k_arr):
        # u must keep h_{k+1} > 0
        u_min_feas = -(hk + f0 * dt) / (G * dt) + 0.01
        def cost(u):
            hk1 = h_next(hk, u, f0)
            if hk1 <= 1e-12:
                return 1e20
            return 0.5 * u**2 - mu_bar * np.log(hk1)
        lb = max(u_min_feas, -10)
        res = minimize_scalar(cost, bounds=(lb, 300), method='bounded')
        hk1_check = h_next(hk, res.x, f0)
        if hk1_check > 1e-10:
            out[i] = res.x / G
    return out


# 4. Exponential: 0.5*u^2 + w*exp(-h_{k+1}/L)
def exponential_multiplier(h_k_arr, f0=f0_default, L=L_default):
    out = np.zeros_like(h_k_arr)
    for i, hk in enumerate(h_k_arr):
        def cost(u):
            hk1 = h_next(hk, u, f0)
            return 0.5 * u**2 + w * np.exp(-hk1 / L)
        res = minimize_scalar(cost, bounds=(-10, 300), method='bounded')
        out[i] = res.x / G
    return out


# ============ Compute ============
f0 = f0_default
mu_cbf = cbf_multiplier(h_range, f0)
lam_quad = quadratic_multiplier(h_range, f0)
lam_log = log_barrier_multiplier(h_range, f0, mu_bar=1.0)
lam_exp = exponential_multiplier(h_range, f0, L=0.5)

print(f"CBF mu range:    [{np.min(mu_cbf):.4f}, {np.max(mu_cbf):.4f}]")
print(f"Quad lambda:     [{np.min(lam_quad):.4f}, {np.max(lam_quad):.4f}]")
print(f"Log lambda:      [{np.nanmin(lam_log):.4f}, {np.nanmax(lam_log):.4f}]")
print(f"Exp lambda:      [{np.min(lam_exp):.4f}, {np.max(lam_exp):.4f}]")


# ============ Figure 1: All four ============
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

ax.plot(h_range, mu_cbf, 'b-', lw=2.5, label=r'CBF: $\mu^*$ (explicit, $C^0$ kink)')
ax.plot(h_range, lam_quad, 'r--', lw=2.5, label=r'Quadratic: $\lambda_q$ (zero for $h_{k+1}>0$)')
ax.plot(h_range, lam_log, 'g-.', lw=2, label=r'Log barrier: $\lambda_{\log}$ (requires $h_{k+1}>0$ for all solver iterates)')
ax.plot(h_range, lam_exp, 'm-', lw=2.5, label=r'Exponential: $\lambda$ ($L=0.5$)')

ax.axvline(x=0, color='k', lw=0.8, ls=':', alpha=0.5)
ax.axhline(y=0, color='k', lw=0.5, alpha=0.3)

h_kink = -f0 / gamma
ax.axvline(x=h_kink, color='blue', lw=0.5, ls='--', alpha=0.4)
ax.annotate(f'CBF activates\n$h_k = {h_kink:.0f}$',
            xy=(h_kink, 0.3), xytext=(h_kink+1.5, 6),
            fontsize=9, arrowprops=dict(arrowstyle='->', color='blue'), color='blue')

ax.annotate('unsafe\n$h<0$', xy=(-2, 0.5), fontsize=11, ha='center', color='red', alpha=0.7)
ax.annotate('safe\n$h>0$', xy=(6, 0.5), fontsize=11, ha='center', color='green', alpha=0.7)

ax.set_xlabel(r'Safety margin $h_k$', fontsize=13)
ax.set_ylabel(r'Effective multiplier $\lambda$ (braking intensity)', fontsize=13)
ax.set_title(r'How each safety formulation responds to the safety margin', fontsize=14)
ax.legend(fontsize=10, loc='upper right')
ax.set_xlim(-12, 8)
ymax = max(np.nanmax(mu_cbf), np.nanmax(lam_quad), np.nanmax(lam_exp)) * 1.05
ax.set_ylim(-1, ymax)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig1_lambda_comparison.png', dpi=200, bbox_inches='tight')
plt.savefig('fig1_lambda_comparison.pdf', bbox_inches='tight')
print("Saved fig1")


# ============ Figure 2: L sweep ============
fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))

L_values = [2.0, 1.0, 0.5, 0.2, 0.1]
colors = ['#ffaaaa', '#ff6666', '#cc3333', '#990000', '#550000']

ax2.plot(h_range, mu_cbf, 'b-', lw=2.5, label=r'CBF: $\mu^*$', zorder=10)
for Lv, col in zip(L_values, colors):
    lam = exponential_multiplier(h_range, f0, L=Lv)
    ax2.plot(h_range, lam, '-', color=col, lw=1.8, label=rf'Exp $L={Lv}$')

ax2.axvline(x=0, color='k', lw=0.8, ls=':', alpha=0.5)
ax2.axhline(y=0, color='k', lw=0.5, alpha=0.3)
ax2.set_xlabel(r'Safety margin $h_k$', fontsize=13)
ax2.set_ylabel(r'Effective multiplier $\lambda$', fontsize=13)
ax2.set_title(r'Sharpness $L$: smaller $\to$ closer to CBF, always smooth', fontsize=14)
ax2.legend(fontsize=10, loc='upper right')
ax2.set_xlim(-3, 8)
ax2.set_ylim(-1, 28)
ax2.grid(True, alpha=0.3)

ax2.annotate(r'$L \to 0$: approaches CBF', xy=(1.5, 3), xytext=(3, 12),
             fontsize=11, ha='center',
             arrowprops=dict(arrowstyle='->', color='black'), color='black')
ax2.annotate('anticipation:\nalways $>0$ here', xy=(3, 0.3), xytext=(5, 6),
             fontsize=10, ha='center',
             arrowprops=dict(arrowstyle='->', color='purple'), color='purple')

plt.tight_layout()
plt.savefig('fig2_exponential_L_sweep.png', dpi=200, bbox_inches='tight')
plt.savefig('fig2_exponential_L_sweep.pdf', bbox_inches='tight')
print("Saved fig2")


# ============ Figure 3: Hessian ============
fig3, ax3 = plt.subplots(1, 1, figsize=(10, 6))

h_hess = np.linspace(-3, 5, 500)

phi2_quad = np.where(h_hess < 0, 2*w, 0.0)
hess_quad = 1 + (G*dt)**2 * phi2_quad

mask_log = h_hess > 0.02
h_log_v = h_hess[mask_log]
hess_log = 1 + (G*dt)**2 * (1.0 / h_log_v**2)

Lp = 0.5
hess_exp = 1 + (G*dt)**2 * (w/Lp**2) * np.exp(-h_hess/Lp)

ax3.plot(h_hess, hess_quad, 'r--', lw=2.5, label=r'Quadratic: jumps at $h=0$')
ax3.plot(h_log_v, hess_log, 'g-.', lw=2, label=r'Log barrier: $\to\infty$ as $h\to 0^+$')
ax3.plot(h_hess, hess_exp, 'm-', lw=2.5, label=rf'Exponential ($L={Lp}$): smooth')
ax3.axhline(y=1, color='gray', lw=0.8, ls=':', alpha=0.5, label=r'Base ($\frac{1}{2}u^2$ only)')
ax3.axvline(x=0, color='k', lw=0.8, ls=':', alpha=0.5)

ax3.set_xlabel(r'Safety margin $h$', fontsize=13)
ax3.set_ylabel(r'Hessian $\partial^2 J/\partial u^2$', fontsize=13)
ax3.set_title('Hessian behaviour determines solver compatibility', fontsize=14)
ax3.legend(fontsize=10, loc='upper right')
ax3.set_xlim(-3, 5)
ax3.set_ylim(0, 30)
ax3.grid(True, alpha=0.3)

ax3.annotate('Quadratic:\njumps here', xy=(0.05, 1.5), xytext=(-2.2, 15),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='red'), color='red')
ax3.annotate('Log barrier:\nblows up', xy=(0.08, 28), xytext=(1.2, 27),
             fontsize=10, arrowprops=dict(arrowstyle='->', color='darkgreen'), color='darkgreen')

plt.tight_layout()
plt.savefig('fig3_hessian_comparison.png', dpi=200, bbox_inches='tight')
plt.savefig('fig3_hessian_comparison.pdf', bbox_inches='tight')
print("Saved fig3")


# ============ Figure 4: Zoom anticipation + recovery ============
fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(12, 5))

# Left: safe region zoom
h_safe = np.linspace(0.01, 5, 200)
ax4a.plot(h_safe, cbf_multiplier(h_safe, f0), 'b-', lw=2.5, label='CBF')
ax4a.plot(h_safe, quadratic_multiplier(h_safe, f0), 'r--', lw=2.5, label='Quadratic')
ax4a.plot(h_safe, log_barrier_multiplier(h_safe, f0, mu_bar=1.0), 'g-.', lw=2, label='Log barrier')
ax4a.plot(h_safe, exponential_multiplier(h_safe, f0, L=0.5), 'm-', lw=2.5, label='Exponential')

ax4a.set_xlabel(r'$h_k$', fontsize=13)
ax4a.set_ylabel(r'$\lambda$', fontsize=13)
ax4a.set_title('Safe region: anticipation', fontsize=13)
ax4a.legend(fontsize=9)
ax4a.set_xlim(0, 5)
ax4a.set_ylim(-0.1, 5)
ax4a.grid(True, alpha=0.3)
ax4a.axhline(y=0, color='k', lw=0.5, alpha=0.3)

ax4a.annotate('CBF & Quadratic:\n$\\lambda = 0$', xy=(3, 0.02), xytext=(3.5, 2.5),
              fontsize=10, ha='center',
              arrowprops=dict(arrowstyle='->', color='red'), color='red')
ax4a.annotate('Exponential:\n$\\lambda > 0$', xy=(2.5, 0.2), xytext=(4.2, 1.2),
              fontsize=10, ha='center',
              arrowprops=dict(arrowstyle='->', color='purple'), color='purple')

# Right: violation region zoom
h_unsafe = np.linspace(-3, -0.01, 200)
mu_uns = cbf_multiplier(h_unsafe, f0)
lq_uns = quadratic_multiplier(h_unsafe, f0)
le_uns = exponential_multiplier(h_unsafe, f0, L=0.5)

ax4b.plot(h_unsafe, mu_uns, 'b-', lw=2.5, label='CBF (linear)')
ax4b.plot(h_unsafe, lq_uns, 'r--', lw=2.5, label='Quadratic')
ax4b.plot(h_unsafe, le_uns, 'm-', lw=2.5, label='Exponential')

ax4b.set_xlabel(r'$h_k$', fontsize=13)
ax4b.set_ylabel(r'$\lambda$', fontsize=13)
ax4b.set_title('Violation region: recovery strength', fontsize=13)
ax4b.legend(fontsize=9, loc='upper right')
ax4b.set_xlim(-3, 0)
ax4b.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig4_anticipation_recovery_zoom.png', dpi=200, bbox_inches='tight')
plt.savefig('fig4_anticipation_recovery_zoom.pdf', bbox_inches='tight')
print("Saved fig4")

plt.show()
print("\nDone.")
