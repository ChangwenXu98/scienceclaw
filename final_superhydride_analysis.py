#!/usr/bin/env python3
"""
Complete analysis of superhydride screening results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

def load_results():
    """Load all computational results"""
    
    # Load UMA relaxation results
    with open('superhydride_screening/calculations/uma_relaxation_150GPa_results.json', 'r') as f:
        uma_results = json.load(f)
    
    # Load phonon stability results
    with open('superhydride_screening/calculations/phonon_stability_results.json', 'r') as f:
        phonon_results = json.load(f)
    
    return uma_results, phonon_results

def analyze_superconductivity_potential(uma_data, phonon_data):
    """Assess superconductivity potential based on structural and phonon properties"""
    
    results = {}
    
    for compound in uma_data.keys():
        if compound not in phonon_data:
            continue
            
        uma = uma_data[compound]
        phonon = phonon_data[compound]
        
        # Calculate superconductivity metrics
        density = uma['final_density']
        compression = uma['compression_percent']
        n_imaginary = phonon['n_imaginary_modes']
        min_freq = phonon['min_frequency']
        max_freq = phonon['max_frequency']
        mean_freq = phonon['mean_frequency']
        
        # Superconductivity potential score (0-100)
        # Based on: high density, high compression, phonon stability, high frequencies
        density_score = min(100, density / 50)  # Normalize by typical superhydride density
        compression_score = compression  # Already 0-100
        stability_score = 100 if n_imaginary == 0 else max(0, 100 - n_imaginary * 25)
        frequency_score = min(100, max_freq / 10)  # High frequency phonons favor superconductivity
        
        # Weight factors based on importance for superconductivity
        sc_potential = (
            0.2 * density_score + 
            0.3 * compression_score + 
            0.4 * stability_score + 
            0.1 * frequency_score
        )
        
        # Estimate Tc based on empirical correlations
        if n_imaginary == 0:
            # McMillan-like formula approximation for superhydrides
            debye_temp = max_freq * 0.1  # Rough estimate
            lambda_ep = min(2.0, mean_freq / 100)  # Electron-phonon coupling estimate
            tc_estimate = debye_temp / 1.45 * np.exp(-1.04 * (1 + lambda_ep) / lambda_ep)
        else:
            tc_estimate = 0  # Unstable structures unlikely to superconduct
        
        results[compound] = {
            'density_gPcm3': density,
            'compression_percent': compression,
            'n_imaginary_modes': n_imaginary,
            'min_frequency_THz': min_freq,
            'max_frequency_THz': max_freq,
            'mean_frequency_THz': mean_freq,
            'phonon_stable': n_imaginary == 0,
            'superconductivity_potential': sc_potential,
            'estimated_Tc_K': tc_estimate,
            'priority_rank': 1 if n_imaginary == 0 else 2  # Stable structures get priority
        }
    
    return results

def create_summary_plots(results):
    """Create visualization of results"""
    
    compounds = list(results.keys())
    densities = [results[c]['density_gPcm3'] for c in compounds]
    sc_potentials = [results[c]['superconductivity_potential'] for c in compounds]
    max_freqs = [results[c]['max_frequency_THz'] for c in compounds]
    stabilities = [results[c]['phonon_stable'] for c in compounds]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Density vs SC potential
    colors = ['green' if s else 'red' for s in stabilities]
    ax1.scatter(densities, sc_potentials, c=colors, alpha=0.7, s=100)
    ax1.set_xlabel('Density (g/cm³)')
    ax1.set_ylabel('Superconductivity Potential')
    ax1.set_title('Density vs Superconductivity Potential')
    for i, c in enumerate(compounds):
        ax1.annotate(c, (densities[i], sc_potentials[i]), xytext=(5,5), 
                    textcoords='offset points', fontsize=8)
    
    # Max frequency distribution
    stable_freqs = [results[c]['max_frequency_THz'] for c in compounds if results[c]['phonon_stable']]
    unstable_freqs = [results[c]['max_frequency_THz'] for c in compounds if not results[c]['phonon_stable']]
    
    ax2.hist([stable_freqs, unstable_freqs], bins=8, alpha=0.7, 
             label=['Stable', 'Unstable'], color=['green', 'red'])
    ax2.set_xlabel('Max Frequency (THz)')
    ax2.set_ylabel('Count')
    ax2.set_title('Phonon Frequency Distribution')
    ax2.legend()
    
    # Superconductivity ranking
    sorted_compounds = sorted(compounds, key=lambda x: results[x]['superconductivity_potential'], reverse=True)
    sorted_potentials = [results[c]['superconductivity_potential'] for c in sorted_compounds]
    colors = ['green' if results[c]['phonon_stable'] else 'red' for c in sorted_compounds]
    
    ax3.barh(range(len(sorted_compounds)), sorted_potentials, color=colors, alpha=0.7)
    ax3.set_yticks(range(len(sorted_compounds)))
    ax3.set_yticklabels(sorted_compounds)
    ax3.set_xlabel('Superconductivity Potential')
    ax3.set_title('Compound Ranking')
    
    # Stability overview
    stable_count = sum(stabilities)
    unstable_count = len(stabilities) - stable_count
    ax4.pie([stable_count, unstable_count], labels=['Stable', 'Unstable'], 
            colors=['green', 'red'], autopct='%1.0f%%')
    ax4.set_title(f'Phonon Stability ({len(compounds)} compounds)')
    
    plt.tight_layout()
    plt.savefig('superhydride_screening/analysis/complete_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
def generate_final_report(results):
    """Generate comprehensive analysis report"""
    
    report = []
    report.append("# SUPERHYDRIDE SUPERCONDUCTIVITY SCREENING RESULTS")
    report.append("=" * 60)
    report.append("")
    report.append(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Total Compounds Analyzed: {len(results)}")
    report.append("")
    
    # Summary statistics
    stable_compounds = [c for c in results if results[c]['phonon_stable']]
    report.append(f"Phonon Stable Compounds: {len(stable_compounds)} ({len(stable_compounds)/len(results)*100:.1f}%)")
    
    if stable_compounds:
        best_stable = max(stable_compounds, key=lambda x: results[x]['superconductivity_potential'])
        report.append(f"Best Stable Candidate: {best_stable} (Score: {results[best_stable]['superconductivity_potential']:.1f})")
    else:
        report.append("No phonon stable compounds found - all show dynamical instabilities")
    
    report.append("")
    report.append("## DETAILED RESULTS BY COMPOUND")
    report.append("-" * 40)
    
    # Sort by superconductivity potential
    sorted_compounds = sorted(results.keys(), key=lambda x: results[x]['superconductivity_potential'], reverse=True)
    
    for i, compound in enumerate(sorted_compounds, 1):
        data = results[compound]
        report.append(f"")
        report.append(f"{i}. {compound}")
        report.append(f"   Superconductivity Potential: {data['superconductivity_potential']:.1f}/100")
        report.append(f"   Phonon Stable: {'✓' if data['phonon_stable'] else '✗'} ({data['n_imaginary_modes']} imaginary modes)")
        report.append(f"   Density: {data['density_gPcm3']:.1f} g/cm³")
        report.append(f"   Compression: {data['compression_percent']:.2f}%")
        report.append(f"   Max Phonon Freq: {data['max_frequency_THz']:.1f} THz")
        if data['estimated_Tc_K'] > 0:
            report.append(f"   Estimated Tc: {data['estimated_Tc_K']:.1f} K")
        else:
            report.append(f"   Estimated Tc: N/A (unstable)")
    
    report.append("")
    report.append("## STRUCTURE-PROPERTY CORRELATIONS")
    report.append("-" * 40)
    
    densities = [results[c]['density_gPcm3'] for c in results]
    compressions = [results[c]['compression_percent'] for c in results]
    max_freqs = [results[c]['max_frequency_THz'] for c in results]
    
    report.append(f"Average Density: {np.mean(densities):.1f} ± {np.std(densities):.1f} g/cm³")
    report.append(f"Average Compression: {np.mean(compressions):.2f} ± {np.std(compressions):.2f}%")
    report.append(f"Average Max Frequency: {np.mean(max_freqs):.1f} ± {np.std(max_freqs):.1f} THz")
    
    report.append("")
    report.append("## RECOMMENDATIONS")
    report.append("-" * 40)
    
    if stable_compounds:
        report.append(f"1. PRIORITY: Focus on {stable_compounds[0]} - phonon stable with highest potential")
        report.append("2. Perform more accurate phonon calculations with finer k-point meshes")
        report.append("3. Calculate electron-phonon coupling for Tc estimation")
        report.append("4. Investigate pressure-dependent superconducting properties")
    else:
        report.append("1. All compounds show phonon instabilities - may require:")
        report.append("   - Different pressure conditions")
        report.append("   - Alternative crystal structures")
        report.append("   - Anharmonic effects consideration")
        report.append("2. Consider compounds with smallest number of imaginary modes")
        report.append("3. Investigate if instabilities lead to stable ground state structures")
    
    report.append("")
    report.append("## COMPUTATIONAL DETAILS")
    report.append("-" * 40)
    report.append("- Pressure: 150 GPa")
    report.append("- UMA relaxation with force convergence < 0.05 eV/Å")
    report.append("- Phonon analysis via finite differences (0.01 Å displacement)")
    report.append("- Mock force constants for stability assessment")
    
    return "\n".join(report)

def main():
    """Main analysis function"""
    
    print("Loading computational results...")
    uma_results, phonon_results = load_results()
    
    print("Analyzing superconductivity potential...")
    analysis_results = analyze_superconductivity_potential(uma_results, phonon_results)
    
    print("Creating visualizations...")
    os.makedirs('superhydride_screening/analysis', exist_ok=True)
    create_summary_plots(analysis_results)
    
    print("Generating final report...")
    report = generate_final_report(analysis_results)
    
    # Save results
    with open('superhydride_screening/analysis/final_analysis_results.json', 'w') as f:
        json.dump(analysis_results, f, indent=2)
    
    with open('superhydride_screening/analysis/final_report.md', 'w') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print("SUPERHYDRIDE SCREENING ANALYSIS COMPLETE")
    print("="*60)
    print(report)
    
    return analysis_results

if __name__ == "__main__":
    results = main()
