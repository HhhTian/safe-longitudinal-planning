"""
CBF vs Exponential vs Quadratic Penalty: Effective Multiplier Comparison
=========================================================================
This script generates a publication-quality figure comparing the effective
Lagrange multiplier λ(h) from three safety-constrained control formulations:

  1. CBF-QP (hard constraint):   u* = μ* · G,  where μ* = max(0, -γh / G²)
  2. Quadratic penalty:          u* = λ(h) · G, where λ = wΔt · max(0, -h)
  3. Exponential penalty:        u* = λ(h) · G, where λ = (wΔt/L) · exp(-h/L)

Smoothness comparison:
  - CBF:        C⁰ discontinuous (kink at h=0)
  - Quadratic:  C⁰ continuous, C¹ discontinuous (1st deriv kink at h=0)
  - Exponential: C∞ (smooth everywhere)

Reference: Bertsekas, "Nonlinear Programming", Section 4.2.5 —
           The Exponential Method of Multipliers.

Usage:
    python cbf_vs_exp_vs_quad_multiplier.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ──────────────────────────────────────────────
# Parameters (adjustable)
# ──────────────────────────────────────────────
w = 2.0        # penalty weight
G = 1.0        # control sensitivity ∂h/∂u
gamma = 0.5    # CBF decay rate
dt = 0.1       # time step

L_values = [0.5, 1.0, 2.0]   # exponential sharpness parameters
w_quad = 2.0                  # quadratic penalty weight

# ──────────────────────────────────────────────
# Compute multiplier functions
# ──────────────────────────────────────────────
h = np.linspace(-3.0, 4.0, 700)

# 1. CBF-QP: μ* = max(0, -γh / G²)
mu_cbf = np.maximum(0, -gamma * h / (G ** 2))

# 2. Quadratic penalty: P(h) = (w_q/2) * max(0, -h)^2
#    => dP/dh = -w_q * max(0, -h)
#    => effective λ = w_q * Δt * max(0, -h)
lambda_quad = w_quad * dt * np.maximum(0, -h)

# 3. Exponential penalty: λ = (w·Δt / L) · exp(-h / L)
lambda_exp = {}
for L in L_values:
    lambda_exp[L] = (w * dt / L) * np.exp(-h / L)

# ──────────────────────────────────────────────
# Figure setup
# ──────────────────────────────────────────────
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9.5,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={'width_ratios': [1, 1]})

# ══════════════════════════════════════════════
# LEFT PANEL: Multiplier λ(h)
# ══════════════════════════════════════════════
ax = axes[0]

# CBF
ax.plot(h, mu_cbf, color='#2B5C9E', linewidth=2.5,
        label=r'CBF: $\mu^* = \max(0,\,-\gamma h / G^2)$', zorder=5)

# Quadratic
ax.plot(h, lambda_quad, color='#7B3FA0', linewidth=2.2, linestyle='-',
        label=r'Quad: $\lambda = w\Delta t \cdot \max(0, -h)$', zorder=4)

# Exponential
exp_colors = {0.5: '#D85A30', 1.0: '#D4950F', 2.0: '#4A8C3F'}
exp_styles = {0.5: '-', 1.0: '--', 2.0: '-.'}

for L in L_values:
    lam = np.clip(lambda_exp[L], 0, 15)
    ax.plot(h, lam, color=exp_colors[L], linewidth=1.8, linestyle=exp_styles[L],
            label=rf'Exp: $L={L}$', zorder=3)

# Constraint boundary
ax.axvline(x=0, color='gray', linewidth=1.0, linestyle=':', alpha=0.6, zorder=1)
ax.annotate('constraint\nboundary $h=0$', xy=(0.02, 7.5),
            xytext=(0.5, 8.5), fontsize=8.5, color='gray', ha='left', va='top',
            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# Region labels
ax.text(-1.8, 0.3, 'unsafe\n$h < 0$', fontsize=9, color='#999999', ha='center', va='bottom')
ax.text(2.8, 0.3, 'safe\n$h > 0$', fontsize=9, color='#999999', ha='center', va='bottom')
ax.axvspan(-3.0, 0, alpha=0.04, color='red', zorder=0)

ax.set_xlim(-3.0, 4.0)
ax.set_ylim(0, 10.0)
ax.set_xlabel(r'Safety margin $h(x)$')
ax.set_ylabel(r'Effective multiplier $\lambda(h)$')
ax.set_title(r'(a) Multiplier $\lambda(h)$: scaling factor in $u^* = \lambda \cdot G$',
             fontweight='medium', fontsize=12, pad=10)
ax.legend(loc='upper right', framealpha=0.9, edgecolor='#cccccc', fontsize=8.5)
ax.grid(True, alpha=0.15)

# Parameter box
param_text = rf"$w={w}$,  $G={G}$,  $\gamma={gamma}$,  $\Delta t={dt}$"
ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
        fontsize=8, va='top', ha='left', color='#888888')

# ══════════════════════════════════════════════
# RIGHT PANEL: Derivative dλ/dh (smoothness)
# ══════════════════════════════════════════════
ax2 = axes[1]

# CBF: dμ/dh = -γ/G² for h<0, 0 for h>0, undefined at h=0
dmu_cbf = np.where(h < -0.01, -gamma / (G**2), np.where(h > 0.01, 0, np.nan))

# Quadratic: dλ/dh = -w*Δt for h<0, 0 for h>0
dlam_quad = np.where(h < -0.01, -w_quad * dt, np.where(h > 0.01, 0, np.nan))

# Exponential: dλ/dh = -(w*Δt/L²) * exp(-h/L)  (continuous everywhere)
dlam_exp = {}
for L in L_values:
    dlam_exp[L] = -(w * dt / L**2) * np.exp(-h / L)

# Plot derivatives
ax2.plot(h, dmu_cbf, color='#2B5C9E', linewidth=2.5,
         label=r'CBF: $d\mu^*/dh$', zorder=5)
ax2.plot(h, dlam_quad, color='#7B3FA0', linewidth=2.2,
         label=r'Quad: $d\lambda/dh$', zorder=4)

for L in L_values:
    dl = np.clip(dlam_exp[L], -5, 0.5)
    ax2.plot(h, dl, color=exp_colors[L], linewidth=1.8, linestyle=exp_styles[L],
             label=rf'Exp: $L={L}$', zorder=3)

# Mark the discontinuities at h=0
ax2.plot(0, 0, 'o', color='#2B5C9E', markersize=5, zorder=6)
ax2.plot(0, -gamma/(G**2), 'o', color='#2B5C9E', markersize=5, fillstyle='none',
         markeredgewidth=1.5, zorder=6)
ax2.annotate(r'CBF: jump $\Delta = \gamma/G^2$',
             xy=(0, -gamma/(G**2)), xytext=(0.6, -0.8),
             fontsize=8, color='#2B5C9E',
             arrowprops=dict(arrowstyle='->', color='#2B5C9E', lw=0.8))

ax2.plot(0, 0, 's', color='#7B3FA0', markersize=5, zorder=6)
ax2.plot(0, -w_quad*dt, 's', color='#7B3FA0', markersize=5, fillstyle='none',
         markeredgewidth=1.5, zorder=6)
ax2.annotate(r'Quad: jump $\Delta = w\Delta t$',
             xy=(0, -w_quad*dt), xytext=(0.6, -0.45),
             fontsize=8, color='#7B3FA0',
             arrowprops=dict(arrowstyle='->', color='#7B3FA0', lw=0.8))

# Constraint boundary
ax2.axvline(x=0, color='gray', linewidth=1.0, linestyle=':', alpha=0.6, zorder=1)
ax2.axhline(y=0, color='gray', linewidth=0.5, alpha=0.4, zorder=0)
ax2.axvspan(-3.0, 0, alpha=0.04, color='red', zorder=0)

ax2.set_xlim(-3.0, 4.0)
ax2.set_ylim(-3.0, 0.5)
ax2.set_xlabel(r'Safety margin $h(x)$')
ax2.set_ylabel(r'$d\lambda / dh$')
ax2.set_title(r'(b) Derivative $d\lambda/dh$: smoothness at boundary',
              fontweight='medium', fontsize=12, pad=10)
ax2.legend(loc='lower right', framealpha=0.9, edgecolor='#cccccc', fontsize=8.5)
ax2.grid(True, alpha=0.15)

# Smoothness annotation box
smooth_text = (
    r"CBF: $C^0$ discontinuous (derivative jumps)" "\n"
    r"Quadratic: $C^0$ continuous, $C^1$ discontinuous" "\n"
    r"Exponential: $C^\infty$ smooth everywhere"
)
ax2.text(0.98, 0.98, smooth_text, transform=ax2.transAxes,
         fontsize=8, va='top', ha='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='#F5F5F0',
                   edgecolor='#CCCCCC', alpha=0.9))

# ──────────────────────────────────────────────
# Global title
# ──────────────────────────────────────────────
fig.suptitle(
    r'Effective multiplier comparison: $u^* = \lambda(h) \cdot G$',
    fontsize=14, fontweight='medium', y=1.02
)

plt.tight_layout()
plt.savefig('cbf_vs_exp_vs_quad_multiplier.png', dpi=300, bbox_inches='tight')
plt.savefig('cbf_vs_exp_vs_quad_multiplier.pdf', bbox_inches='tight')
print("Saved: cbf_vs_exp_vs_quad_multiplier.png / .pdf")
plt.show()