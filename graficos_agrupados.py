import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter


h = 4.135667696e-15 
c = 2.99792458e8    

def calculate_wavelength(energy_gev):
    """Calculate wavelength in nm from gamma energy in GeV."""
    if energy_gev <= 0:
        return None
    energy_ev = energy_gev * 1e9 
    wavelength_m = (h * c) / energy_ev
    return wavelength_m * 1e9 

def load_and_process_files(directory="../simulation_results_20250513_115132/"):
    """Load all .hit files and organize data by material and energy."""
    data = defaultdict(lambda: defaultdict(list))
    
    for filename in os.listdir(directory):
        if filename.endswith('.hit'):
            try:
               
                parts = filename.split('-')
                material = parts[0].upper() 
                energy = parts[1]
                seed = parts[2].split('.')[0]
                
                filepath = os.path.join(directory, filename)
                gamma_energies = []
                
                with open(filepath, 'r') as f:
                    for line in f:
                        fields = line.strip().split()
                        if len(fields) >= 12 and fields[11].lower() == 'gamma':
                            try:
                                e_kin_gev = float(fields[10])
                                if e_kin_gev > 0:
                                    gamma_energies.append(e_kin_gev)
                            except (ValueError, IndexError):
                                continue
                
                if gamma_energies:
                    wavelengths = [w for w in (calculate_wavelength(e) for e in gamma_energies) if w is not None]
                    if wavelengths:
                        data[energy][material].extend(wavelengths)
            
            except (IndexError, ValueError) as e:
                print(f"Error processing {filename}: {str(e)}")
                continue
    
    return data

def plot_grouped_histograms(data, output_dir="plots"):
    """Generate grouped histograms by energy with all materials."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    image_files = []
    colors = {'AIR': 'dodgerblue', 'CO2': 'limegreen', 'CH4': 'tomato'}
    hatch_patterns = {'AIR': '///', 'CO2': 'xxx', 'CH4': '...'}
    
    for energy, materials_data in sorted(data.items(), key=lambda x: float(x[0])):
        plt.figure(figsize=(12, 8))
        plt.style.use('seaborn')
        
       
        plot_data = []
        labels = []
        for material, wavelengths in materials_data.items():
            if wavelengths:
                plot_data.append(wavelengths)
                labels.append(f"{material} (N={len(wavelengths)})")
        
        if not plot_data:
            continue
        
       
        n_bins = 50
        _, bins, _ = plt.hist(
            plot_data,
            bins=n_bins,
            color=[colors.get(m, 'gray') for m in materials_data.keys()],
            alpha=0.7,
            edgecolor='black',
            linewidth=0.5,
            label=labels,
            stacked=False,
            density=True,
            hatch=[hatch_patterns.get(m, '') for m in materials_data.keys()]
        )
        
        plt.xlabel('Wavelength (nm)', fontsize=12)
        plt.ylabel('Normalized Frequency', fontsize=12)
        plt.title(f'Gamma Wavelength Distribution at {energy} GeV', fontsize=14, pad=20)
        plt.legend(fontsize=10, framealpha=1)
        plt.grid(True, linestyle=':', alpha=0.7)
        
       
        if any(w > 10*min(w for w in plot_data[0]) for w in plot_data[0]):
            plt.yscale('log')
        
       
        all_wavelengths = [w for wl in plot_data for w in wl]
        plt.xlim(0.9*min(all_wavelengths), 1.1*max(all_wavelengths))
        
       
        output_file = os.path.join(output_dir, f'gamma_wavelengths_{energy}GeV.png')
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        image_files.append(output_file)
    
    return image_files

def plot_statistics_comparison(data, output_dir="plots"):
    """Plot comparison of statistical measures across energies."""
    plt.figure(figsize=(14, 8))
    plt.style.use('seaborn')
    
    colors = {'AIR': 'dodgerblue', 'CO2': 'limegreen', 'CH4': 'tomato'}
    markers = {'AIR': 'o', 'CO2': 's', 'CH4': '^'}
    
   
    energies = sorted(data.keys(), key=float)
    materials = sorted(set(mat for energy_data in data.values() for mat in energy_data.keys()))
    
   
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
    
    for material in materials:
        means = []
        stds = []
        counts = []
        
        for energy in energies:
            if material in data[energy] and data[energy][material]:
                wavelengths = data[energy][material]
                means.append(np.mean(wavelengths))
                stds.append(np.std(wavelengths))
                counts.append(len(wavelengths))
            else:
                means.append(np.nan)
                stds.append(np.nan)
                counts.append(0)
        
       
        ax1.errorbar(
            [float(e) for e in energies],
            means,
            yerr=stds,
            marker=markers[material],
            color=colors[material],
            linestyle='-',
            markersize=8,
            capsize=5,
            label=material,
            linewidth=2
        )
        
       
        ax2.plot(
            [float(e) for e in energies],
            counts,
            marker=markers[material],
            color=colors[material],
            linestyle='--',
            markersize=8,
            label=material,
            linewidth=2
        )
    
   
    ax1.set_title('Mean Gamma Wavelength by Material and Energy', fontsize=14)
    ax1.set_ylabel('Mean Wavelength (nm)', fontsize=12)
    ax1.legend(fontsize=10, framealpha=1)
    ax1.grid(True, linestyle=':', alpha=0.7)
    ax1.set_yscale('log')
    
   
    ax2.set_title('Gamma Counts by Material and Energy', fontsize=14)
    ax2.set_xlabel('Energy (GeV)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.legend(fontsize=10, framealpha=1)
    ax2.grid(True, linestyle=':', alpha=0.7)
    ax2.set_yscale('log')
    
   
    plt.xticks([float(e) for e in energies])
    
   
    output_file = os.path.join(output_dir, 'gamma_statistics_comparison.png')
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_file

def create_report(image_files, output_pdf="gamma_analysis_report.pdf"):
    """Create PDF report from generated plots."""
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter
    
    for image_file in image_files:
        if os.path.exists(image_file):
           
            c.setFont("Helvetica-Bold", 16)
            title = os.path.basename(image_file).replace('.png', '').replace('_', ' ')
            c.drawCentredString(width/2, height-30, title)
            
           
            img_width = width - 2*72
            img_height = height - 2*72
            c.drawImage(image_file, 72, 72, width=img_width, height=img_height, preserveAspectRatio=True)
            c.showPage()
    
    c.save()
    print(f"\nPDF report generated: {output_pdf}")

def main():
    print("Starting gamma wavelength analysis...")
    
   
    data = load_and_process_files()
    if not data:
        print("No valid data found in .hit files!")
        return
    
   
    print("\nGenerating plots...")
    histograms = plot_grouped_histograms(data)
    stats_plot = plot_statistics_comparison(data)
    
   
    print("\nCreating PDF report...")
    create_report(histograms + [stats_plot])
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()