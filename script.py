import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import plotly.graph_objects as go
from plotly.subplots import make_subplots

h = 4.135667696e-15  
c = 2.99792458e8     
MeV_to_eV = 1e-6

def calculate_wavelengthasd(energy_MeV):
    if energy_MeV <= 0:
        return None
    wavelength_m = (h * c) / (energy_MeV * MeV_to_eV)
    return wavelength_m * 1e9  

h_J = 6.62607015e-34  # (J·s)
c = 2.99792458e8     # (m/s)
eV_to_J = 1.60218e-19  # 1 eV = 1.60218e-19 J

def calculate_wavelength(energy_MeV):
    energy_eV = energy_MeV * 1e6  # MeV -> eV
    energy_J = energy_eV * eV_to_J  # eV -> Joules
    wavelength_m = (h_J * c) / energy_J
    return wavelength_m * 1e9  # m -> nm

def load_and_group_files(directory):
    """Load all .hit files and group them by energy."""
    data = defaultdict(lambda: defaultdict(list))
    depth_energy_data = defaultdict(lambda: defaultdict(list))  
    
    for filename in os.listdir(directory):
        print("Processing file: ", filename)
        if filename.endswith('.hit'):
            try:
                parts = filename.split('-')
                material = parts[0]
                energy = parts[1]
                seed = parts[2].split('.')[0]
                
                filepath = os.path.join(directory, filename)
                gamma_energies = []
                processed_particles = set() 
                
                with open(filepath, 'r') as f:
                    for line in f:
                        fields = line.strip().split()
                        if len(fields) >= 12 and fields[11] == 'gamma':
                            try:
                                particle_key = (fields[0], fields[1], fields[2])
                                
                                if particle_key not in processed_particles:
                                    processed_particles.add(particle_key)
                                    e_kin_ev = float(fields[10])
                                    z_pos = float(fields[5])  
                                    
                                    if e_kin_ev > 0:
                                        gamma_energies.append(e_kin_ev)
                                        
                                        depth_energy_data[energy][material].append((z_pos, e_kin_ev))
                            except (ValueError, IndexError):
                                continue
                
                if gamma_energies:
                    wavelengths = [w for w in (calculate_wavelength(e) for e in gamma_energies) if w is not None]
                    if wavelengths:
                        data[energy][material].extend(wavelengths)
            
            except (IndexError, ValueError) as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    return data, depth_energy_data  

def plot_grouped_histograms(data, output_dir="analysis_results"):
    """Generate grouped histograms by energy with adjusted bins."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = []
    colors = {'Air': 'blue', 'CO2': 'green', 'CH4': 'red'}
    
    for energy, materials_data in data.items():
        plt.figure(figsize=(12, 7))
        
        
        all_wavelengths = [w for mat_data in materials_data.values() for w in mat_data]
        min_wl = max(0, np.min(all_wavelengths) - 0.1)
        max_wl = np.max(all_wavelengths) + 0.1
        bins = np.linspace(min_wl, max_wl, 10000)  
        
        for material, wavelengths in materials_data.items():
            plt.hist(
                wavelengths,
                bins=bins,
                color=colors.get(material, 'gray'),
                alpha=0.6,
                edgecolor='black',
                linewidth=0.5,
                label=f'{material} (N={len(wavelengths)})'
            )
        
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Count')
        plt.title(f'Gamma Wavelength Distribution at {energy} GeV')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_file = os.path.join(output_dir, f'gamma_wavelengths_{energy}GeV.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        image_files.append(output_file)
    
    return image_files

def plot_statistics_comparison(data, output_dir="analysis_results"):
    """Plot comparison of mean wavelengths."""
    plt.figure(figsize=(12, 7))
    
    colors = {'Air': 'blue', 'CO2': 'green', 'CH4': 'red'}
    markers = {'Air': 'o', 'CO2': 's', 'CH4': '^'}
    
    
    energies = sorted(data.keys(), key=float)
    materials = sorted(set(mat for energy_data in data.values() for mat in energy_data.keys()))
    
    for material in materials:
        means = []
        stds = []
        
        for energy in energies:
            if material in data[energy]:
                wavelengths = data[energy][material]
                means.append(np.mean(wavelengths))
                stds.append(np.std(wavelengths))
            else:
                means.append(np.nan)
                stds.append(np.nan)
        
        plt.errorbar(
            [float(e) for e in energies],
            means,
            yerr=stds,
            marker=markers.get(material, 'o'),
            color=colors.get(material, 'gray'),
            linestyle='-',
            markersize=8,
            capsize=5,
            label=material
        )
    
    plt.title('Mean Gamma Wavelength by Material and Energy')
    plt.xlabel('Energy (GeV)')
    plt.ylabel('Mean Wavelength (nm)')
    plt.xticks([float(e) for e in energies])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    output_file = os.path.join(output_dir, 'wavelength_comparison.png')
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    return output_file

def create_interactive_plot(data, output_dir="analysis_results"):
    """Create interactive plot with zoom capability."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    colors = {'Air': 'blue', 'CO2': 'green', 'CH4': 'red'}
    
    for energy, materials_data in data.items():
        fig = make_subplots(rows=2, cols=1, 
                           vertical_spacing=0.1,
                           subplot_titles=(f"Full View - {energy} GeV", "Zoomed View"))
        
        
        all_wavelengths = [w for mat_data in materials_data.values() for w in mat_data]
        for material, wavelengths in materials_data.items():
            fig.add_trace(
                go.Histogram(
                    x=wavelengths,
                    name=material,
                    marker_color=colors.get(material, 'gray'),
                    opacity=0.3,
                    nbinsx=10000,
                    showlegend=True
                ),
                row=1, col=1
            )
            
        
        fig.update_layout(
            title_text=f"Gamma Wavelength Distribution at {energy} GeV",
            height=800,
            barmode='overlay'
        )
        
        
        q1, q3 = np.percentile(all_wavelengths, [5, 95])
        fig.update_xaxes(range=[q1, q3], row=2, col=1)
        
        output_file = os.path.join(output_dir, f'interactive_gamma_wavelengths_{energy}GeV.html')
        fig.write_html(output_file)
        
    return output_file

def plot_energy_vs_depth(depth_energy_data, output_dir="analysis_results", convert_z_to_mm=False):
    """
    Plot particle energy vs depth (Z position) with proper unit handling.
    
    Parameters:
        depth_energy_data: Dictionary containing (z_pos_cm, energy_MeV) data
        output_dir: Output directory for plots
        convert_z_to_mm: If True, converts Z position from cm to mm in the plot
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = []
    colors = {'Air': 'blue', 'CO2': 'green', 'CH4': 'red'}
    
    for energy, materials_data in depth_energy_data.items():
        plt.figure(figsize=(12, 7))
        
        for material, data_points in materials_data.items():
            if data_points:
                
                z_pos_cm, energies_MeV = zip(*data_points)
                
                
                z_pos = np.array(z_pos_cm) * 10 if convert_z_to_mm else np.array(z_pos_cm)
                z_unit = 'mm' if convert_z_to_mm else 'cm'
                energies_eV = np.array(energies_MeV) * 1e6  
                
                plt.scatter(
                    z_pos,
                    energies_eV,
                    color=colors.get(material, 'gray'),
                    alpha=0.6,
                    label=f'{material} (N={len(energies_eV)})',
                    s=10
                )
        
        plt.xlabel(f'Depth (Z position) [{z_unit}]')
        plt.ylabel('Energy [eV]')
        plt.title(f'Particle Energy vs Depth at {energy} GeV (Primary Beam)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        
        plt.yscale('log')
        
        output_file = os.path.join(output_dir, f'energy_vs_depth_{energy}GeV_{z_unit}.png')

        
        plt.annotate(f'Data from {output_file}', 
                    xy=(0.05, 0.05), xycoords='axes fraction',
                    fontsize=8, color='gray', alpha=0.7)        
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        image_files.append(output_file)
    
    return image_files

def main():
    sim_dirs = [d for d in os.listdir('.') if d.startswith('simulation_results_')]
    if not sim_dirs:
        print("No simulation results directory found!")
        return
    
    latest_dir = sorted(sim_dirs)[-1]
    directory = os.path.join('.', latest_dir)
    print(f"Analyzing files in: {directory}")
    
    grouped_data, depth_energy_data = load_and_group_files(directory)  
    if not grouped_data:
        print("No valid data found!")
        return
    
    
    # depth_plots = plot_energy_vs_depth(depth_energy_data)
    
    # # Static histograms
    # histograms = plot_grouped_histograms(grouped_data)
    
    # # Statistical comparison
    # comparison = plot_statistics_comparison(grouped_data)
    
    # Interactive plots
    interactive_plots = create_interactive_plot(grouped_data)

    print("\nAnalysis complete! Generated plots:")
    # print(f"- Energy vs depth plots: {len(depth_plots)} files")
    # print(f"- Grouped histograms by energy: {len(histograms)} files")
    # print(f"- Statistical comparison: {comparison}")
    print(f"- Interactive plots: {interactive_plots}")


if __name__ == "__main__":
    main()