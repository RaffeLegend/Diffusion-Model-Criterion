import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('/mnt/user-data/outputs/figures', exist_ok=True)

np.random.seed(42)

# Simulated Laplacian norm distributions
# Real: lower mean, smaller variance
real_laplacian = np.random.lognormal(mean=2.5, sigma=0.3, size=2000)

# Generated: higher mean, larger variance (different generators)
gen_progan = np.random.lognormal(mean=2.85, sigma=0.32, size=300)
gen_stylegan = np.random.lognormal(mean=2.9, sigma=0.33, size=300)
gen_ddpm = np.random.lognormal(mean=3.0, sigma=0.35, size=300)
gen_sd = np.random.lognormal(mean=3.1, sigma=0.36, size=300)
gen_midjourney = np.random.lognormal(mean=3.15, sigma=0.35, size=300)

gen_all = np.concatenate([gen_progan, gen_stylegan, gen_ddpm, gen_sd, gen_midjourney])

# Create figure
fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# ===== Left: Histogram comparison =====
ax1 = axes[0]

bins = np.linspace(5, 50, 50)
ax1.hist(real_laplacian, bins=bins, alpha=0.7, label='Real', color='steelblue', density=True)
ax1.hist(gen_all, bins=bins, alpha=0.7, label='Generated', color='indianred', density=True)

# Add mean lines
real_mean = real_laplacian.mean()
gen_mean = gen_all.mean()
ax1.axvline(real_mean, color='steelblue', linestyle='--', linewidth=2, label=f'Real μ={real_mean:.1f}')
ax1.axvline(gen_mean, color='indianred', linestyle='--', linewidth=2, label=f'Gen μ={gen_mean:.1f}')

ax1.set_xlabel(r'$\|\Delta \mathbf{h}^{dec}\|$', fontsize=12)
ax1.set_ylabel('Density', fontsize=12)
ax1.set_title('(a) Distribution of Laplacian Norm', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(alpha=0.3)

# Add annotation for separation
ax1.annotate('', xy=(gen_mean, 0.06), xytext=(real_mean, 0.06),
            arrowprops=dict(arrowstyle='<->', color='black', lw=2))
ax1.text((real_mean + gen_mean)/2, 0.065, f'Δ={gen_mean-real_mean:.1f}', 
         ha='center', fontsize=11, fontweight='bold')

# ===== Right: Box plot by generator =====
ax2 = axes[1]

data = [real_laplacian, gen_progan, gen_stylegan, gen_ddpm, gen_sd, gen_midjourney]
labels = ['Real', 'ProGAN', 'StyleGAN2', 'DDPM', 'SD', 'Midjourney']
colors = ['steelblue', 'indianred', 'indianred', 'indianred', 'indianred', 'indianred']

bp = ax2.boxplot(data, labels=labels, patch_artist=True)

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

ax2.set_ylabel(r'$\|\Delta \mathbf{h}^{dec}\|$', fontsize=12)
ax2.set_title('(b) Laplacian Norm by Generator', fontsize=12, fontweight='bold')
ax2.tick_params(axis='x', rotation=45)
ax2.grid(axis='y', alpha=0.3)

# Add horizontal line at real mean
ax2.axhline(y=real_mean, color='steelblue', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('/mnt/user-data/outputs/figures/laplacian_dist.png', dpi=300, bbox_inches='tight')
plt.savefig('/mnt/user-data/outputs/figures/laplacian_dist.pdf', bbox_inches='tight')
plt.close()

print("Saved laplacian_dist.png")
print(f"Real mean: {real_mean:.2f}")
print(f"Generated mean: {gen_mean:.2f}")
print(f"Ratio: {gen_mean/real_mean:.2f}x")