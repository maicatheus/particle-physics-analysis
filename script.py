import os
import numpy as np
import plotly.express as px
import pandas as pd
from collections import defaultdict
import plotly.graph_objects as go 
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

def create_output_directory(base_path="."):
    """Cria a estrutura de diretórios para os resultados da análise"""
    output_dir = os.path.join(base_path, "analysis_results")
    os.makedirs(output_dir, exist_ok=True)
    
    
    dirs = {
        'histograms': os.path.join(output_dir, "histograms"),
        '3d_plots': os.path.join(output_dir, "3d_plots"),
        'full_3d_plots': os.path.join(output_dir, "full_3d_plots"),
        'heatmaps': os.path.join(output_dir, "heatmaps"),
    }
    
    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    return dirs


def load_and_group_files_enhanced(directory):
    """Carrega os arquivos .hit e agrupa os dados por energia e material"""
    energy_material_data = defaultdict(lambda: defaultdict(list))
    
    for filename in os.listdir(directory):
        if not filename.endswith('.hit'):
            continue
            
        print(f"Processando arquivo: {filename}")
        try:
            parts = filename.split('-')
            material = parts[0]
            energy = parts[1]
            
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
                                if e_kin_MeV > 0:
                                    energy_material_data[energy][material].append(e_kin_MeV)
                        except (ValueError, IndexError) as e:
                            print(f"Erro ao processar linha: {line.strip()}")
                            continue
        except (IndexError, ValueError) as e:
            print(f"Erro ao processar {filename}: {str(e)}")
            continue
            
    return energy_material_data


def create_interactive_plot_histogram(data, bins=1000, output_dir="histograms"):
    """Cria histogramas interativos de contagem de partículas por energia"""
    colors = {'Air': 'blue', 'CO2': 'green', 'CH4': 'red'}
    
    for energy, materials_data in data.items():
        fig = make_subplots(rows=2, cols=1, 
                          vertical_spacing=0.1)
        
        all_energies = []
        for material, entries in materials_data.items():
            energies = [entry[3] for entry in entries]  
            all_energies.extend(energies)
            
            fig.add_trace(
                go.Histogram(
                    x=energies,
                    name=material,
                    marker=dict(
                        color='rgba(0,0,0,0)', 
                        line=dict(
                            color=colors.get(material, 'gray'),
                            width=1
                        )
                    ),
                    opacity=0.7,
                    nbinsx=bins,
                    showlegend=True
                ),
                row=1, col=1
            )
        
        
        fig.update_layout(
            title_text=f"Distribuição de Energia - E0 = {energy} GeV",
            height=800,
            barmode='overlay',
            xaxis_title="Energia (MeV)",
            yaxis_title="Número de Partículas"
        )
        
        
        if all_energies:
            q1, q3 = np.percentile(all_energies, [5, 95])
            fig.update_xaxes(range=[q1, q3], row=2, col=1)
            
            
            for material, entries in materials_data.items():
                energies = [entry[3] for entry in entries]
                fig.add_trace(
                    go.Histogram(
                        x=energies,
                        name=material,
                        marker_color=colors.get(material, 'gray'),
                        opacity=0.5,
                        nbinsx=bins,
                        showlegend=False
                    ),
                    row=2, col=1
                )
        
        
        safe_energy = "".join(c for c in energy if c.isalnum() or c in (' ', '_')).rstrip()
        output_file = os.path.join(output_dir, f"histograma_{safe_energy}.html")
        fig.write_html(output_file)
    
    return output_dir


def create_3d_plots(position_energy_data, output_dir, z_ranges=None):
    """Cria todos os tipos de gráficos 3D organizados por material"""
    
    if z_ranges is None:
        z_ranges = [
            (-10000, -9800),     
            (-5000, -4800),      
            (0, 200),            
            (4800, 5000),        
            (9800, 10000)        
        ]
    
    
    for energy, materials in position_energy_data.items():
        for material, entries in materials.items():
            
            safe_material = "".join(c for c in material if c.isalnum() or c in (' ', '_')).rstrip()
            material_dir = os.path.join(output_dir, safe_material)
            os.makedirs(material_dir, exist_ok=True)
            
            
            create_combined_3d_plot(entries, material, energy, z_ranges, material_dir)
            
            
            create_individual_layer_plots(entries, material, energy, z_ranges, material_dir)

def create_full_3d_plot(position_energy_data, output_dir, grid_size=0.5, max_points=5000):
    """
    Cria um gráfico 3D completo mostrando todas as partículas de -10000 a 10000
    com pontos agrupados para reduzir a quantidade total.
    """
    full_3d_dir = os.path.join(output_dir, "full_3d_plots")
    os.makedirs(full_3d_dir, exist_ok=True)
    
    for energy, materials in position_energy_data.items():
        for material, entries in materials.items():
            
            grid = defaultdict(list)
            
            for entry in entries:
                x_m, y_m, z_m, e_kin = entry
                grid_x = round(x_m / grid_size) * grid_size
                grid_y = round(y_m / grid_size) * grid_size
                grid_z = round(z_m / grid_size) * grid_size
                grid_key = (grid_x, grid_y, grid_z)
                grid[grid_key].append(e_kin)
            
            if not grid:
                print(f"Nenhuma partícula encontrada para {material}-{energy}")
                continue
            
            
            df_list = []
            for (x, y, z), energies in grid.items():
                df_list.append({
                    'X': x,
                    'Y': y,
                    'Z': z,
                    'Energy': np.mean(energies),
                    'Count': len(energies),
                    'Material': material,
                    'Energy_Group': energy
                })
            
            if len(df_list) > max_points:
                df_list = pd.DataFrame(df_list).sample(max_points).to_dict('records')
                print(f"Reduzindo pontos de {len(df_list)} para {max_points} em {material}-{energy}")
            
            df = pd.DataFrame(df_list)
            
            
            fig = go.Figure()
            
            
            fig.add_trace(
                go.Scatter3d(
                    x=df['X'],
                    y=df['Y'],
                    z=df['Z'],
                    mode='markers',
                    marker=dict(
                        size=df['Count']/df['Count'].max()*10 + 3,  
                        color=df['Energy'],
                        colorscale='thermal',
                        colorbar=dict(title='Energia Média (MeV)'),
                        opacity=0.7,
                        line=dict(width=0)
                    ),
                    text=[f"Partículas: {c}<br>Energia média: {e:.2f} MeV" 
                          for c, e in zip(df['Count'], df['Energy'])],
                    hoverinfo='text',
                    name=material
                )
            )
            
            
            fig.update_layout(
                title=f"{material} - {energy} GeV",
                scene=dict(
                    xaxis_title='X (m)',
                    yaxis_title='Y (m)',
                    zaxis_title='Z (m)',
                    aspectmode='manual',
                    aspectratio=dict(x=1, y=1, z=2),
                    zaxis=dict(range=[10000, -10000]),  
                    camera=dict(
                        eye=dict(x=1.5, y=1.5, z=-0.5),  
                        up=dict(x=0, y=0, z=1)           
                    )
                ),
                margin=dict(l=0, r=0, b=0, t=30)
            )
            
            
            safe_material = "".join(c for c in material if c.isalnum() or c in (' ', '_')).rstrip()
            safe_energy = "".join(c for c in energy if c.isalnum() or c in (' ', '_')).rstrip()
            output_file = os.path.join(full_3d_dir, f"full_3d_grouped_{safe_material}_{safe_energy}.html")
            fig.write_html(output_file)
            print(f"Gráfico 3D agrupado salvo em: {output_file}")


def create_combined_3d_plot(entries, material, energy, z_ranges, output_dir):
    """Cria um gráfico 3D com todos os layers combinados"""
    df_list = []
    layers_present = set()
    
    for layer_idx, (z_min, z_max) in enumerate(z_ranges, start=1):
        for entry in entries:
            x_m, y_m, z_m, e_kin = entry
            if z_min <= z_m <= z_max:
                df_list.append({
                    'X': x_m, 'Y': y_m, 'Z': z_m,
                    'Energy': e_kin,
                    'Layer': f'Layer {layer_idx}',
                    'Z_Range': f'{z_min:.1f}m a {z_max:.1f}m'
                })
                layers_present.add(layer_idx)
    
    if not df_list:
        print(f"Nenhuma partícula encontrada para {material}-{energy}")
        return
    
    df = pd.DataFrame(df_list)
    
    fig = px.scatter_3d(
        df,
        x='X', y='Y', z='Z',
        color='Energy',
        title=f"{material} - {energy} GeV",
        labels={'X': 'X (m)', 'Y': 'Y (m)', 'Z': 'Z (m)', 'Energy': 'Energia (MeV)'},
        color_continuous_scale='thermal',
        hover_data=['Layer', 'Z_Range']
    )
    
    fig.update_layout(
        scene=dict(aspectmode='data'),
        coloraxis_colorbar=dict(title='Energia (MeV)')
    )
    
    
    fig.update_traces(
        marker=dict(size=4, opacity=0.7, line=dict(width=0)),
        selector=dict(mode='markers')
    )
    
    
    safe_energy = "".join(c for c in energy if c.isalnum() or c in (' ', '_')).rstrip()
    output_file = os.path.join(output_dir, f"ALL_LAYERS_{safe_energy}_3d.html")
    fig.write_html(output_file)
    print(f"Gráfico 3D combinado salvo em: {output_file}")


def create_individual_layer_plots(entries, material, energy, z_ranges, output_dir):
    """Cria gráficos 3D com barras de energia mais espessas para cada layer"""
    for layer_idx, (z_min, z_max) in enumerate(z_ranges, start=1):
        
        sampled_entries = []
        grid_size = 0.01  
        
        
        grid = {}
        for entry in entries:
            x_m, y_m, z_m, e_kin = entry
            if z_min <= z_m <= z_max:
                
                grid_x = round(x_m / grid_size) * grid_size
                grid_y = round(y_m / grid_size) * grid_size
                grid_key = (grid_x, grid_y)
                
                if grid_key not in grid:
                    grid[grid_key] = {
                        'x': grid_x,
                        'y': grid_y,
                        'energies': [],
                        'count': 0
                    }
                grid[grid_key]['energies'].append(e_kin)
                grid[grid_key]['count'] += 1
        
        
        bar_data = []
        for key in grid:
            avg_energy = np.mean(grid[key]['energies'])
            bar_data.append({
                'X': grid[key]['x'],
                'Y': grid[key]['y'],
                'Energy': avg_energy,
                'Count': grid[key]['count']
            })
        
        if not bar_data:
            print(f"Nenhuma partícula encontrada para {material}-{energy} no Layer {layer_idx}")
            continue
        
        
        max_bars = 1000
        if len(bar_data) > max_bars:
            
            bar_data = pd.DataFrame(bar_data).sample(max_bars).to_dict('records')
            print(f"Amostra reduzida para {max_bars} barras no Layer {layer_idx}")
        
        
        fig = go.Figure()
        
        
        for bar in bar_data:
            fig.add_trace(go.Scatter3d(
                x=[bar['X'], bar['X']],
                y=[bar['Y'], bar['Y']],
                z=[0, bar['Energy']],
                mode='lines',
                line=dict(
                    width=20 + 15 * min(bar['Count']/10, 5),  
                    color=bar['Energy'],
                    colorscale='thermal',
                    cmin=0,  
                    cmax=max(b['Energy'] for b in bar_data)  
                ),
                hoverinfo='text',
                text=f"Pos: ({bar['X']:.1f}, {bar['Y']:.1f})<br>Energia média: {bar['Energy']:.2f} MeV<br>Partículas: {bar['Count']}",
                showlegend=False
            ))
        
        
        fig.update_layout(
            title=dict(
                text=f"{material} ({energy} GeV) - Layer {layer_idx}<br>Z: {z_min:.1f}m a {z_max:.1f}m | {len(bar_data)} barras",
                y=0.95,
                x=0.5
            ),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Energia Média (MeV)',
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.5),
                camera=dict(
                    eye=dict(x=1.2, y=1.2, z=0.6),
                    up=dict(x=0, y=0, z=1)  
                )
            ),
            margin=dict(l=0, r=0, b=0, t=100),
            coloraxis_colorbar=dict(title='Energia (MeV)')
        )
        
        
        safe_material = "".join(c for c in material if c.isalnum() or c in (' ', '_')).rstrip()
        safe_energy = "".join(c for c in energy if c.isalnum() or c in (' ', '_')).rstrip()
        png_file = os.path.join(output_dir, f"{safe_material}_{safe_energy}_layer{layer_idx}.png")
        
        
        try:
            fig.write_image(
                png_file,
                width=1200,
                height=800,
                scale=2,
                engine="kaleido"
            )
            print(f"Gráfico 3D salvo em: {png_file}")
        except Exception as e:
            print(f"Erro ao salvar gráfico: {str(e)}")
            
            try:
                fig.write_image(
                    png_file,
                    width=800,
                    height=600,
                    scale=1,
                    engine="kaleido"
                )
                print(f"Gráfico salvo com configurações reduzidas")
            except:
                print(f"Falha ao salvar o gráfico para {material}-{energy} layer {layer_idx}")


def create_heatmaps(position_energy_data, output_dir, z_ranges=None):
    """Cria mapas de calor 2D interativos (HTML) e estáticos (PNG) para cada material, energia e layer"""
    
    if z_ranges is None:
        z_ranges = [
            (-10000, -9800),     
            (-5000, -4800),      
            (0, 200),            
            (4800, 5000),        
            (9800, 10000)        
        ]
    
    for energy, materials in position_energy_data.items():
        for material, entries in materials.items():
            for layer_idx, (z_min, z_max) in enumerate(z_ranges, start=1):
                df_list = []
                
                
                for entry in entries:
                    x_m, y_m, z_m, e_kin = entry
                    if z_min <= z_m <= z_max:
                        df_list.append({
                            'X': x_m,
                            'Y': y_m,
                            'Z': z_m,
                            'Energy': e_kin,
                            'Material': material,
                            'Energy_Group': energy
                        })
                
                if not df_list:
                    print(f"Nenhuma partícula encontrada para {material}-{energy} no layer {layer_idx} ({z_min:.2f}m a {z_max:.2f}m)")
                    continue
                
                df = pd.DataFrame(df_list)
                
                
                fig = px.density_heatmap(
                    df,
                    x='X',
                    y='Y',
                    z='Energy',
                    nbinsx=100,
                    nbinsy=100,
                    title=f"{material} ({energy} GeV) - Layer {layer_idx} (Z: {z_min:.2f}m a {z_max:.2f}m)",
                    labels={
                        'X': 'Posição X (m)',
                        'Y': 'Posição Y (m)',
                        'Energy': 'Energia (MeV)'
                    },
                    color_continuous_scale='viridis',
                    hover_data=['Material', 'Energy_Group', 'Z']
                )
                
                fig.update_layout(
                    coloraxis_colorbar=dict(
                        title='Energia (MeV)',
                        thicknessmode='pixels',
                        thickness=20,
                        lenmode='pixels',
                        len=300,
                        yanchor='top',
                        y=1,
                        ticks='outside'
                    ),
                    xaxis_title='Posição X (m)',
                    yaxis_title='Posição Y (m)',
                    hovermode='closest',
                    width=1200,
                    height=800
                )
                
                
                safe_material = "".join(c for c in material if c.isalnum() or c in (' ', '_')).rstrip()
                safe_energy = "".join(c for c in energy if c.isalnum() or c in (' ', '_')).rstrip()
                
                
                
                
                
                
                
                png_file = os.path.join(output_dir, f"{safe_material}-{safe_energy}-layer{layer_idx}_heat.png")
                try:
                    fig.write_image(
                        png_file,
                        width=1200,
                        height=800,
                        scale=2,
                        engine="kaleido"
                    )
                    print(f"Mapa de calor PNG salvo em: {png_file}")
                except Exception as e:
                    print(f"Erro ao salvar PNG: {str(e)}")
                    
                    try:
                        fig.write_image(
                            png_file,
                            width=800,
                            height=600,
                            scale=1,
                            engine="kaleido"
                        )
                        print(f"PNG salvo com configurações reduzidas")
                    except Exception as e2:
                        print(f"Falha ao salvar o PNG: {str(e2)}")


def load_and_group_files_with_positions(directory):
    """Carrega os arquivos .hit convertendo posições para metros"""
    position_energy_data = defaultdict(lambda: defaultdict(list))
    
    for filename in os.listdir(directory):
        if not filename.endswith('.hit'):
            continue
            
        print(f"Processando arquivo: {filename}")
        try:
            parts = filename.split('-')
            material = parts[0]
            energy = parts[1]
            
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
                                
                                x_pos = float(fields[3]) / 100
                                y_pos = float(fields[4]) / 100
                                z_pos = float(fields[5]) / 100
                                if e_kin_MeV > 0:
                                    position_energy_data[energy][material].append((
                                        x_pos, y_pos, z_pos, e_kin_MeV
                                    ))
                        except (ValueError, IndexError) as e:
                            print(f"Erro ao processar linha: {line.strip()}")
                            continue
        except (IndexError, ValueError) as e:
            print(f"Erro ao processar {filename}: {str(e)}")
            continue
            
    return position_energy_data



def plot_max_energy_comparison(position_energy_data, output_dir="analysis_results"):
    """Cria gráficos cartesianos da energia máxima por camada"""
    import plotly.express as px
    
    z_ranges = [
        (-10000, -9800),     
        (-5000, -4800),      
        (0, 200),            
        (4800, 5000),        
        (9800, 10000)        
    ]
    
    
    plot_dir = os.path.join(output_dir, "energy_comparison_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    
    all_data = []
    
    for energy, materials_data in position_energy_data.items():
        for layer_idx, (z_min, z_max) in enumerate(z_ranges, start=1):
            for material, entries in materials_data.items():
                photons = [entry for entry in entries if z_min <= entry[2] <= z_max]
                if not photons:
                    continue
                    
                energies = [photon[3] for photon in photons]
                max_energy = max(energies)
                
                all_data.append({
                    "Material": material,
                    "Energia Máxima (MeV)": max_energy,
                    "Camada": layer_idx,
                    "Z Range": f"{z_min} a {z_max} m",
                    "Energia Inicial": energy
                })
    
    if not all_data:
        print("Nenhum dado encontrado para gráficos de energia máxima")
        return
    
    df = pd.DataFrame(all_data)
    
    
    for energy in df['Energia Inicial'].unique():
        df_energy = df[df['Energia Inicial'] == energy]
        
        fig = px.line(df_energy,
                     x="Camada",
                     y="Energia Máxima (MeV)",
                     color="Material",
                     markers=True,
                     title=f"Energia Máxima dos Fótons por Camada - Energia Inicial {energy} GeV",
                     labels={"Camada": "Camada", "Energia Máxima (MeV)": "Energia Máxima (MeV)"},
                     height=600)
        
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, len(z_ranges)+1)),
                ticktext=[f"Camada {i}" for i in range(1, len(z_ranges)+1)]
            ),
            legend_title_text='Material'
        )
        
        
        safe_energy = "".join(c for c in energy if c.isalnum() or c in (' ', '_')).rstrip()
        html_file = os.path.join(plot_dir, f"max_energy_line_{safe_energy}.html")
        png_file = os.path.join(plot_dir, f"max_energy_line_{safe_energy}.png")
        
        fig.write_html(html_file)
        fig.write_image(png_file, width=1000, height=600, scale=2)
        
        print(f"Gráfico de energia máxima para {energy} GeV salvo em: {png_file}")


def plot_photon_count_by_spectral_region(position_energy_data, output_dir="analysis_results"):
    """Gráfico de contagem de fótons por camada, classificados por região espectral (UV, Visível, IV)."""
    import plotly.express as px

    
    spectral_regions = {
        "UV": (3.1e-6, 124e-6),          
        "Visível": (1.65e-6, 3.1e-6),     
        "IV": (0.0012e-6, 1.65e-6)        
    }

    z_ranges = [
        (-10000, -9800),     
        (-5000, -4800),      
        (0, 200),            
        (4800, 5000),        
        (9800, 10000)        
    ]

    
    plot_dir = os.path.join(output_dir, "spectral_region_plots")
    os.makedirs(plot_dir, exist_ok=True)

    
    all_data = []
    total_detected = 0  

    for energy, materials_data in position_energy_data.items():
        for layer_idx, (z_min, z_max) in enumerate(z_ranges, start=1):
            for material, entries in materials_data.items():
                photons = [entry for entry in entries if z_min <= entry[2] <= z_max]
                if not photons:
                    continue

                
                energies = [photon[3] for photon in photons]

                
                counts = {}
                for region, (e_min, e_max) in spectral_regions.items():
                    counts[region] = sum(e_min <= energy <= e_max for energy in energies)
                    total_detected += counts[region]

                total_photons = len(photons)

                for region, count in counts.items():
                    all_data.append({
                        "Material": material,
                        "Região Espectral": region,
                        "Contagem de Fótons": count,
                        "Porcentagem": (count / total_photons) * 100 if total_photons > 0 else 0,
                        "Camada": layer_idx,
                        "Z Range": f"{z_min} a {z_max} m",
                        "Energia Inicial": energy
                    })

    
    if total_detected == 0:
        print("\nAVISO: Nenhum fóton foi detectado nas faixas UV/Visível/IV.")
        print("Motivo provável: Seus dados estão na faixa de MeV (alta energia),")
        print("enquanto UV/Visível/IV estão na faixa de eV (1 MeV = 1.000.000 eV).")
        
        
        fig = px.bar(title="Nenhum fóton detectado nas faixas UV/Visível/IV<br>"
                          "Seus dados estão em MeV (alta energia), enquanto<br>"
                          "UV (3.1-124 eV), Visível (1.65-3.1 eV), IV (0.0012-1.65 eV)")
        fig.update_layout(annotations=[dict(
            text="Dados incompatíveis com faixas espectrais",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False
        )])
        
        html_file = os.path.join(plot_dir, "AVISO_spectral_regions.html")
        png_file = os.path.join(plot_dir, "AVISO_spectral_regions.png")
        fig.write_html(html_file)
        fig.write_image(png_file, width=800, height=400)
        return

    df = pd.DataFrame(all_data)

    
    for energy in df['Energia Inicial'].unique():
        df_energy = df[df['Energia Inicial'] == energy]

        for material in df_energy['Material'].unique():
            df_material = df_energy[df_energy['Material'] == material]

            
            fig = px.bar(df_material,
                        x="Camada",
                        y="Contagem de Fótons",
                        color="Região Espectral",
                        title=f"Fótons por Região Espectral - {material} - Energia {energy} GeV<br>"
                              f"(UV: 3.1-124 eV, Visível: 1.65-3.1 eV, IV: 0.0012-1.65 eV)",
                        labels={"Camada": "Camada", "Contagem de Fótons": "Número de Fótons"},
                        height=600,
                        color_discrete_map={"UV": "violet", "Visível": "green", "IV": "red"})

            fig.update_layout(
                xaxis=dict(
                    tickmode='array',
                    tickvals=list(range(1, len(z_ranges)+1)),
                    ticktext=[f"Camada {i}" for i in range(1, len(z_ranges)+1)]
                ),
                barmode='stack',
                legend_title_text='Região Espectral'
            )

            
            safe_energy = "".join(c for c in energy if c.isalnum() or c in (' ', '_')).rstrip()
            safe_material = "".join(c for c in material if c.isalnum() or c in (' ', '_')).rstrip()
            html_file = os.path.join(plot_dir, f"spectral_regions_{safe_material}_{safe_energy}.html")
            png_file = os.path.join(plot_dir, f"spectral_regions_{safe_material}_{safe_energy}.png")

            fig.write_html(html_file)
            fig.write_image(png_file, width=1000, height=600, scale=2)

            print(f"Gráfico para {material} ({energy} GeV) salvo em: {png_file}")

    
    fig_consolidated = px.bar(df,
                             x="Camada",
                             y="Contagem de Fótons",
                             color="Região Espectral",
                             facet_row="Energia Inicial",
                             facet_col="Material",
                             title="Distribuição Espectral de Fótons por Camada, Material e Energia",
                             height=1500,
                             color_discrete_map={"UV": "violet", "Visível": "green", "IV": "red"})

    fig_consolidated.update_layout(
        barmode='stack',
        legend_title_text='Região Espectral'
    )

    html_file = os.path.join(plot_dir, "spectral_regions_consolidated.html")
    png_file = os.path.join(plot_dir, "spectral_regions_consolidated.png")

    fig_consolidated.write_html(html_file)
    fig_consolidated.write_image(png_file, width=1200, height=1500, scale=2)
    
def plot_photon_count_comparison(position_energy_data, output_dir="analysis_results"):
    """Cria gráficos cartesianos da contagem de fótons por camada"""
    import plotly.express as px
    
    z_ranges = [
        (-10000, -9800),     
        (-5000, -4800),      
        (0, 200),            
        (4800, 5000),        
        (9800, 10000)        
    ]
    
    
    plot_dir = os.path.join(output_dir, "photon_count_plots")
    os.makedirs(plot_dir, exist_ok=True)
    
    
    all_data = []
    
    for energy, materials_data in position_energy_data.items():
        for layer_idx, (z_min, z_max) in enumerate(z_ranges, start=1):
            for material, entries in materials_data.items():
                photons = [entry for entry in entries if z_min <= entry[2] <= z_max]
                if not photons:
                    continue
                    
                count = len(photons)
                
                all_data.append({
                    "Material": material,
                    "Número de Fótons": count,
                    "Camada": layer_idx,
                    "Z Range": f"{z_min} a {z_max} m",
                    "Energia Inicial": energy
                })
    
    if not all_data:
        print("Nenhum dado encontrado para gráficos de contagem")
        return
    
    df = pd.DataFrame(all_data)
    
    
    for energy in df['Energia Inicial'].unique():
        df_energy = df[df['Energia Inicial'] == energy]
        
        fig = px.line(df_energy,
                     x="Camada",
                     y="Número de Fótons",
                     color="Material",
                     markers=True,
                     title=f"Contagem de Fótons por Camada - Energia Inicial {energy} GeV",
                     labels={"Camada": "Camada", "Número de Fótons": "Número de Fótons"},
                     height=600)
        
        fig.update_layout(
            xaxis=dict(
                tickmode='array',
                tickvals=list(range(1, len(z_ranges)+1)),
                ticktext=[f"Camada {i}" for i in range(1, len(z_ranges)+1)]
            ),
            legend_title_text='Material',
            yaxis_type="log"  
        )
        
        
        safe_energy = "".join(c for c in energy if c.isalnum() or c in (' ', '_')).rstrip()
        html_file = os.path.join(plot_dir, f"photon_count_line_{safe_energy}.html")
        png_file = os.path.join(plot_dir, f"photon_count_line_{safe_energy}.png")
        
        fig.write_html(html_file)
        fig.write_image(png_file, width=1000, height=600, scale=2)
        
        print(f"Gráfico de contagem para {energy} GeV salvo em: {png_file}")
  
def generate_layer_analysis_report(position_energy_data, output_dir="analysis_results"):
    """Generate simplified photon production reports per layer with wavelength conversion"""
    
    def energy_to_wavelength(energy_mev):
        """Convert MeV to wavelength in picometers (pm)"""
        hc = 1240  # MeV·fm
        return round(hc / (energy_mev * 1000), 2) if energy_mev > 0 else 0
    
    # Layer Z-ranges (in mm)
    z_ranges = [
        (-10000, -9800),  # Layer 1
        (-5000, -4800),   # Layer 2
        (0, 200),         # Layer 3
        (4800, 5000),     # Layer 4
        (9800, 10000)     # Layer 5
    ]
    
    # Create output directory
    report_dir = os.path.join(output_dir, "photon_layer_reports")
    os.makedirs(report_dir, exist_ok=True)
    
    all_reports = []
    
    # Process each energy level
    for energy, materials_data in position_energy_data.items():
        energy_report = {
            "energy": f"{float(energy):.3f} GeV",
            "layers": []
        }
        
        # Analyze each layer
        for layer_idx, (z_min, z_max) in enumerate(z_ranges, start=1):
            layer_data = {
                "layer": layer_idx,
                "z_range": f"{z_min/1000:.1f} to {z_max/1000:.1f} m",
                "materials": [],
                "total_photons": 0
            }
            
            material_stats = []
            
            # Count photons per material
            for material, entries in materials_data.items():
                photons = [e for e in entries if z_min <= e[2] <= z_max]
                if not photons:
                    continue
                    
                energies = [p[3] for p in photons]  # Assuming energy is at index 3
                max_energy = max(energies)
                
                material_stats.append({
                    "material": material,
                    "count": len(photons),
                    "max_energy_mev": max_energy,
                    "wavelength_pm": energy_to_wavelength(max_energy),
                    "mean_energy": sum(energies)/len(energies)
                })
                layer_data["total_photons"] += len(photons)
            
            if material_stats:
                # Sort by highest energy first
                material_stats.sort(key=lambda x: x["max_energy_mev"], reverse=True)
                layer_data["materials"] = material_stats
                layer_data["most_energetic"] = material_stats[0]["material"]
                energy_report["layers"].append(layer_data)
        
        all_reports.append(energy_report)
    
    # Generate HTML report
    html_report = """<!DOCTYPE html>
<html>
<head>
    <title>Photon Layer Analysis with Wavelength</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px }
        .energy-box { background: #f0f8ff; padding: 15px; margin-bottom: 20px; border-radius: 5px }
        .layer-box { background: #f9f9f9; padding: 10px; margin: 10px 0; border-left: 4px solid #4682b4 }
        table { width: 100%; border-collapse: collapse; margin: 10px 0 }
        th, td { padding: 8px; text-align: left; border: 1px solid #ddd }
        th { background-color: #4682b4; color: white }
        .download-btn { background: #4CAF50; color: white; padding: 6px 12px; text-decoration: none; border-radius: 4px }
    </style>
</head>
<body>
    <h1>Photon Production by Layer</h1>
    <p>Includes energy-to-wavelength conversion (hc = 1240 MeV·fm)</p>
"""
    
    for report in all_reports:
        html_report += f"""
    <div class="energy-box">
        <h2>Initial Energy: {report['energy']}</h2>"""
        
        for layer in report["layers"]:
            # Create dataframe for table
            df = pd.DataFrame(layer["materials"])
            df = df[["material", "count", "max_energy_mev", "wavelength_pm", "mean_energy"]]
            df.columns = ["Material", "Count", "Max Energy (MeV)", "Wavelength (pm)", "Mean Energy (MeV)"]
            
            # Save as PNG
            table_id = f"table_{report['energy']}_{layer['layer']}".replace(" ", "_")
            png_path = os.path.join(report_dir, f"{table_id}.png")
            
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.axis('off')
            table = ax.table(
                cellText=df.values,
                colLabels=df.columns,
                cellLoc='center',
                loc='center'
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.2)
            plt.savefig(png_path, bbox_inches='tight', dpi=300)
            plt.close()
            
            html_report += f"""
        <div class="layer-box">
            <h3>Layer {layer['layer']} (Z: {layer['z_range']})</h3>
            <p><b>Total photons:</b> {layer['total_photons']:,}</p>
            <p><b>Most energetic material:</b> {layer['most_energetic']}</p>
            
            <h4>Material Details:</h4>
            {df.to_html(index=False, border=1)}
            
            <a href="{png_path}" download class="download-btn">
                Download as PNG
            </a>
        </div>"""
        
        html_report += "\n    </div>"
    
    html_report += """
</body>
</html>"""
    
    # Save report
    report_path = os.path.join(report_dir, "photon_analysis_report.html")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html_report)
    
    print(f"Report generated at: {report_path}")
    return report_dir


def main():
    sim_dirs = [d for d in os.listdir('.') if d.startswith('simulation_results_')]
    if not sim_dirs:
        print("Nenhum diretório de resultados encontrado!")
        return
    
    latest_dir = sorted(sim_dirs)[-1]
    directory = os.path.join('.', latest_dir)
    print(f"Analisando arquivos em: {directory}")
    
    
    output_dirs = create_output_directory()
    
    
    position_energy_data = load_and_group_files_with_positions(directory)
    
    if not position_energy_data:
        print("Nenhum dado de partícula gamma encontrado!")
        return
    
    
    custom_z_ranges = [
        (-10000, -9800),     
        (-5000, -4800),      
        (0, 200),            
        (4800, 5000),        
        (9800, 10000)        
    ]

    
    # print("\nGerando visualizações...")
    # print("1. Histogramas interativos...")
    # create_interactive_plot_histogram(position_energy_data, 10000, output_dirs['histograms'])
    
    # print("\n2. Gráficos 3D...")
    # create_3d_plots(position_energy_data, output_dirs['3d_plots'], z_ranges=custom_z_ranges)
        
    # print("\n3. Gráfico 3D completo...")
    # create_full_3d_plot(position_energy_data, output_dirs['full_3d_plots'])

    # print("\n3. Mapas de calor...")
    # create_heatmaps(position_energy_data, output_dirs['heatmaps'], z_ranges=custom_z_ranges)
    
    # print("\nAnálise concluída com sucesso!")
    # print(f"Resultados salvos em: {os.path.abspath('analysis_results')}")
    
    # print("\n4. Gerando relatórios de análise por camada...")
    # generate_layer_analysis_report(position_energy_data, output_dirs['histograms'])
    
    # print("\n5. Gerando gráficos comparativos de energia máxima...")
    # plot_max_energy_comparison(position_energy_data, output_dirs['histograms'])
    
    print("\n6. Gerando gráficos comparativos de contagem de fótons...")
    plot_photon_count_comparison(position_energy_data, output_dirs['histograms'])
    
    # print("\n7. Gerando gráficos por região espectral (UV/Visível/IV)...")
    # plot_photon_count_by_spectral_region(position_energy_data, output_dirs['histograms'])
    
if __name__ == "__main__":
    main()