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

def plot_3d_particle_positions(depth_energy_data, output_dir="3d_plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = []
    
    for energy, materials_data in depth_energy_data.items():
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        for material, data_points in materials_data.items():
            if data_points:

                positions = np.array([(p[0], p[1], p[2]) for p in data_points])  # x, y, z
                energies = np.array([p[3] for p in data_points])  # energy
                
                norm = plt.Normalize(min(energies), max(energies))
                colors = plt.cm.viridis(norm(energies))

                ax.scatter(
                    positions[:, 0],  # x
                    positions[:, 1],  # y
                    positions[:, 2],  # z
                    c=colors,
                    marker='o',
                    alpha=0.6,
                    label=material,
                    s=10
                )

        ax.set_xlabel('X Position (cm)')
        ax.set_ylabel('Y Position (cm)')
        ax.set_zlabel('Z Position (cm)')
        ax.set_title(f'3D Particle Positions at {energy} GeV (Color by Energy)')
    
        sm = plt.cm.ScalarMappable(cmap='viridis', norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=ax, label='Energy (MeV)', shrink=0.5, aspect=10)
        
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_file = os.path.join(output_dir, f'3d_positions_{energy}GeV.png')
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()
        image_files.append(output_file)
    
    return image_files

def load_and_group_files_enhanced(directory):
    data = defaultdict(lambda: defaultdict(list))
    position_energy_data = defaultdict(lambda: defaultdict(list))
    
    for filename in os.listdir(directory):
        print("Processing file: ", filename)
        if filename.endswith('.hit'):
            try:
                parts = filename.split('-')
                material = parts[0]
                energy = parts[1]
                seed = parts[2].split('.')[0]
                
                filepath = os.path.join(directory, filename)
                processed_particles = set() 
                
                with open(filepath, 'r') as f:
                    for line in f:
                        fields = line.strip().split()
                        if len(fields) >= 12 and fields[11] == 'gamma':
                            try:
                                particle_key = (fields[0], fields[1], fields[2])
                                
                                if particle_key not in processed_particles:
                                    processed_particles.add(particle_key)
                                    e_kin_MeV = float(fields[10])
                                    x_pos = float(fields[3])  # x position
                                    y_pos = float(fields[4])  # y position
                                    z_pos = float(fields[5])  # z position
                                    
                                    if e_kin_MeV > 0:
                                        wavelengths = calculate_wavelength(e_kin_MeV)
                                        if wavelengths is not None:
                                            data[energy][material].append(wavelengths)
                                        
                                        position_energy_data[energy][material].append((
                                            x_pos, y_pos, z_pos, e_kin_MeV 
                                        ))
                            except (ValueError, IndexError) as e:
                                print(f"Error processing line: {line.strip()}")
                                print(f"Error: {str(e)}")
                                continue
            except (IndexError, ValueError) as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    return data, position_energy_data

def plot_interactive_3d_positions(position_energy_data, output_dir="3d_plots", max_points=50000):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_files = []
    
    MIN_ENERGY = 0.1  # MeV
    MAX_ENERGY = 0.5    # MeV
    
    for energy, materials_data in position_energy_data.items():
        for material, data_points in materials_data.items():
            if not data_points:
                continue
                
            fig = go.Figure()
            
            data_array = np.array(data_points)
            positions = data_array[:, :3]  # x, y, z cm
            energies = data_array[:, 3]    # energy MeV
            
            if len(data_array) > max_points:
                step = len(data_array) // max_points
                data_array = data_array[::step]
                positions = positions[::step]
                energies = energies[::step]
                print(f"Downsampled {material} at {energy} GeV from {len(data_points)} to {len(data_array)} points")
            
            clipped_energies = np.clip(energies, MIN_ENERGY, MAX_ENERGY)
            
            scatter = go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=4,
                    color=clipped_energies,
                    colorscale='Viridis',
                    cmin=MIN_ENERGY,
                    cmax=MAX_ENERGY,
                    colorbar=dict(
                        title='Energy [MeV]',
                        tickvals=np.linspace(MIN_ENERGY, MAX_ENERGY, 5),
                        ticktext=[f"{val:.2f}" for val in np.linspace(MIN_ENERGY, MAX_ENERGY, 5)]
                    ),
                    opacity=0.7,
                    showscale=True
                ),
                name=material,
                text=[f'Material: {material}<br>Energy: {e:.3f} MeV<br>Position: ({x:.1f}, {y:.1f}, {z:.1f}) cm' 
                      for e, x, y, z in zip(energies, positions[:,0], positions[:,1], positions[:,2])],
                hoverinfo='text'
            )
            
            fig.add_trace(scatter)
            
            if np.any(energies < MIN_ENERGY):
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='gray',
                        opacity=0.7
                    ),
                    name=f'Energy < {MIN_ENERGY} MeV',
                    showlegend=True
                ))
            
            if np.any(energies > MAX_ENERGY):
                fig.add_trace(go.Scatter3d(
                    x=[None], y=[None], z=[None],
                    mode='markers',
                    marker=dict(
                        size=8,
                        color='red',
                        opacity=0.7
                    ),
                    name=f'Energy > {MAX_ENERGY} MeV',
                    showlegend=True
                ))
            
            fig.update_layout(
                title=f'3D Particle Positions - {material} at {energy} GeV<br>'
                      f'Color range: {MIN_ENERGY}-{MAX_ENERGY} MeV | Points: {len(data_array)}',
                scene=dict(
                    xaxis_title='X Position (cm)',
                    yaxis_title='Y Position (cm)',
                    zaxis_title='Z Position (cm)',
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                margin=dict(l=0, r=0, b=0, t=50),
                height=800,
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            output_file = os.path.join(output_dir, f'3d_{material}_{energy}GeV.html')
            fig.write_html(output_file)
            output_files.append(output_file)
    
    return output_files


def main():
    sim_dirs = [d for d in os.listdir('.') if d.startswith('simulation_results_')]
    if not sim_dirs:
        print("No simulation results directory found!")
        return
    
    latest_dir = sorted(sim_dirs)[-1]
    directory = os.path.join('.', latest_dir)
    print(f"Analyzing files in: {directory}")
    
    grouped_data, position_energy_data = load_and_group_files_enhanced(directory)
    if not grouped_data:
        print("No valid data found!")
        return
    
    # # Interactive 3D position plots
    interactive_3d_plots = plot_interactive_3d_positions(position_energy_data, max_points=25000)

    # 3D position plots
    # position_plots = plot_3d_particle_positions(position_energy_data)
    
    # # Static histograms
    # histograms = plot_grouped_histograms(grouped_data)
    
    # Statistical comparison
    # comparison = plot_statistics_comparison(grouped_data)
    
    # Interactive wavelength plots
    # interactive_plots = create_interactive_plot(grouped_data)

    print("\nAnalysis complete! Generated plots:")
    print(f"- Interactive 3D position plots: {len(interactive_3d_plots)} files")
    # print(f"- 3D position plots: {len(position_plots)} files")
    # print(f"- Grouped histograms by energy: {len(histograms)} files")
    # print(f"- Statistical comparison: {comparison}")
    # print(f"- Interactive wavelength plots: {interactive_plots}")

if __name__ == "__main__":
    main()