"""
Generate Low-Frequency Similarity Figure
=========================================
Shows that real and generated images have similar low-frequency energy ratios,
while differing in high-frequency energy (γ < ε).

Usage:
    python generate_lf_similarity_fig.py --output_dir figures

This creates a figure showing:
- Bar chart comparing LF energy ratio for Real vs Generated
- The gap γ is small (similar LF structure)
- Contrast with HF energy gap ε (which is large)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def generate_lf_similarity_figure(output_dir='figures'):
    """Generate the low-frequency similarity comparison figure."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Simulated data (replace with real experimental data)
    # These should come from your actual experiments
    
    generators = ['ProGAN', 'StyleGAN2', 'BigGAN', 'DDPM', 'ADM', 'SD', 'Midjourney']
    
    # Low-frequency energy ratio E^low_K / ||f||^2 (should be similar)
    real_lf_ratio = 0.858  # 1 - 0.142 (from HF table)
    gen_lf_ratios = [0.822, 0.817, 0.814, 0.805, 0.811, 0.799, 0.797]
    
    # High-frequency energy ratio E^high_K / ||f||^2 (should differ)
    real_hf_ratio = 0.142
    gen_hf_ratios = [0.178, 0.183, 0.186, 0.195, 0.189, 0.201, 0.203]
    
    # Compute gaps
    lf_gaps = [abs(g - real_lf_ratio) / real_lf_ratio * 100 for g in gen_lf_ratios]
    hf_gaps = [(g - real_hf_ratio) / real_hf_ratio * 100 for g in gen_hf_ratios]
    
    avg_lf_gap = np.mean(lf_gaps)  # This is γ
    avg_hf_gap = np.mean(hf_gaps)  # This is ε
    
    # Create figure with two subplots
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    # ===== Left: Bar chart of LF and HF ratios =====
    ax1 = axes[0]
    x = np.arange(len(generators))
    width = 0.35
    
    # Plot Real as horizontal lines
    ax1.axhline(y=real_lf_ratio, color='steelblue', linestyle='--', linewidth=2, 
                label=f'Real LF ({real_lf_ratio:.3f})')
    ax1.axhline(y=real_hf_ratio, color='indianred', linestyle='--', linewidth=2,
                label=f'Real HF ({real_hf_ratio:.3f})')
    
    # Plot Generated
    bars1 = ax1.bar(x - width/2, gen_lf_ratios, width, label='Gen LF', 
                    color='steelblue', alpha=0.7)
    bars2 = ax1.bar(x + width/2, gen_hf_ratios, width, label='Gen HF', 
                    color='indianred', alpha=0.7)
    
    ax1.set_ylabel('Energy Ratio', fontsize=11)
    ax1.set_xlabel('Generator', fontsize=11)
    ax1.set_title('(a) Energy Ratios by Frequency Band', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(generators, rotation=45, ha='right', fontsize=9)
    ax1.legend(loc='center right', fontsize=9)
    ax1.set_ylim([0, 1.0])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add annotations
    ax1.annotate('', xy=(0.5, real_lf_ratio), xytext=(0.5, np.mean(gen_lf_ratios)),
                arrowprops=dict(arrowstyle='<->', color='steelblue', lw=1.5))
    ax1.text(0.7, (real_lf_ratio + np.mean(gen_lf_ratios))/2, r'$\gamma$', 
             fontsize=12, color='steelblue', fontweight='bold')
    
    # ===== Right: Gap comparison (γ vs ε) =====
    ax2 = axes[1]
    
    categories = ['Low-Freq Gap\n' + r'($\gamma$)', 'High-Freq Gap\n' + r'($\varepsilon$)']
    gaps = [avg_lf_gap, avg_hf_gap]
    colors = ['steelblue', 'indianred']
    
    bars = ax2.bar(categories, gaps, color=colors, alpha=0.8, width=0.5)
    
    # Add value labels
    for bar, gap in zip(bars, gaps):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{gap:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax2.set_ylabel('Relative Gap vs Real (%)', fontsize=11)
    ax2.set_title('(b) Gap Comparison: ' + r'$\gamma \ll \varepsilon$', fontsize=12, fontweight='bold')
    ax2.set_ylim([0, max(gaps) * 1.3])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add annotation showing γ < ε
    ax2.annotate(r'$\gamma < \varepsilon$ ✓', xy=(0.5, max(gaps) * 0.8),
                fontsize=14, ha='center', color='green', fontweight='bold',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    
    # Save
    plt.savefig(os.path.join(output_dir, 'lf_similarity.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'lf_similarity.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved to {output_dir}/lf_similarity.png")
    print(f"\nStatistics:")
    print(f"  Average LF gap (γ): {avg_lf_gap:.2f}%")
    print(f"  Average HF gap (ε): {avg_hf_gap:.2f}%")
    print(f"  γ < ε: {avg_lf_gap < avg_hf_gap}")


def generate_alternative_figure(output_dir='figures'):
    """
    Alternative: Single plot showing both ratios side by side for each generator.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generators = ['Real', 'ProGAN', 'StyleGAN2', 'BigGAN', 'DDPM', 'ADM', 'SD', 'Midjourney']
    
    lf_ratios = [0.858, 0.822, 0.817, 0.814, 0.805, 0.811, 0.799, 0.797]
    hf_ratios = [0.142, 0.178, 0.183, 0.186, 0.195, 0.189, 0.201, 0.203]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    x = np.arange(len(generators))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, lf_ratios, width, label=r'$E^{low}_K / \|f\|^2$', 
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, hf_ratios, width, label=r'$E^{high}_K / \|f\|^2$', 
                   color='indianred', alpha=0.8)
    
    # Highlight Real
    bars1[0].set_edgecolor('black')
    bars1[0].set_linewidth(2)
    bars2[0].set_edgecolor('black')
    bars2[0].set_linewidth(2)
    
    ax.set_ylabel('Energy Ratio', fontsize=12)
    ax.set_xlabel('Source', fontsize=12)
    ax.set_title('Frequency Energy Distribution: Real vs Generated', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(generators, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.set_ylim([0, 1.0])
    ax.grid(axis='y', alpha=0.3)
    
    # Add bracket annotations
    # LF: small variation
    ax.annotate('', xy=(0, 0.858), xytext=(7, 0.797),
                arrowprops=dict(arrowstyle='-', color='steelblue', lw=1, ls='--'))
    ax.text(4, 0.88, r'LF: small gap ($\gamma$)', fontsize=10, color='steelblue', ha='center')
    
    # HF: large variation  
    ax.text(4, 0.22, r'HF: large gap ($\varepsilon$)', fontsize=10, color='indianred', ha='center')
    
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'lf_similarity_alt.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'lf_similarity_alt.pdf'), bbox_inches='tight')
    plt.close()
    
    print(f"Saved alternative figure to {output_dir}/lf_similarity_alt.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='figures')
    args = parser.parse_args()
    
    generate_lf_similarity_figure(args.output_dir)
    generate_alternative_figure(args.output_dir)
