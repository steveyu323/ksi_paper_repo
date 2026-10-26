#!/usr/bin/env python3
"""
Complete Script to Generate All Three Final Visualizations
==========================================================

Input: 260129_ptmvar_kinase_substrate_pind_type_all.csv

Outputs:
- viz1_final_overview_dashboard.png (7-panel dashboard with disease/mutation stats)
- viz2_final_matrix_with_sidebar.png (kinase-substrate matrix with unique pair counts)
- viz4_final.png (pathway networks with phosphosite nodes facing outward)

Author: Claude
Date: January 30, 2026
Version: Final
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from matplotlib.path import Path
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# LOAD DATA
# ============================================================================

def load_data(filepath):
    """Load and prepare the kinase-substrate data"""
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} kinase-substrate-phosphosite entries")
    return df

# ============================================================================
# COLOR SCHEMES (consistent across all figures)
# ============================================================================

TIER_COLORS = {
    'Tier 1': '#2ecc71',  # Green
    'Tier 2': '#f39c12',  # Orange
    'Tier 3': '#e74c3c'   # Red
}

CONFIDENCE_COLORS = {
    'HIGH': '#27ae60',
    'MEDIUM': '#f39c12', 
    'LOW': '#c0392b'
}

NODE_COLORS = {
    'kinase': '#3498db',      # Blue
    'substrate': '#8e44ad',   # Purple
    'pind': '#e74c3c'         # Red
}

# ============================================================================
# FIGURE 1: Overview Dashboard
# ============================================================================

def generate_viz1(df, output_path='viz1_final_overview_dashboard.png'):
    """
    Generate comprehensive 7-panel overview dashboard
    
    Panels:
    1. Tier distribution (pie)
    2. Confidence levels (bar)
    3. Mutation type distribution (bar)
    4. Mutation type by tier (grouped bar)
    5. Top 15 diseases (horizontal bar)
    6. Disease categories by tier (stacked bar)
    7. Top 12 substrates by kinase count (stacked bar)
    """
    
    fig = plt.figure(figsize=(24, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    
    # Panel 1: Tier Distribution Pie Chart
    ax1 = fig.add_subplot(gs[0, 0])
    tier_counts = df['tier'].value_counts()
    colors_pie = [TIER_COLORS[t] for t in tier_counts.index]
    wedges, texts, autotexts = ax1.pie(
        tier_counts.values, 
        labels=tier_counts.index, 
        autopct='%1.1f%%', 
        colors=colors_pie,
        startangle=90, 
        textprops={'fontsize': 11, 'weight': 'bold'}
    )
    ax1.set_title('Tier Distribution (n=110)', fontsize=14, weight='bold', pad=20)
    for autotext in autotexts:
        autotext.set_color('white')
    
    # Panel 2: Consensus Confidence Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    conf_counts = df['consensus_confidence'].value_counts().reindex(['HIGH', 'MEDIUM', 'LOW'])
    bars = ax2.bar(
        conf_counts.index, 
        conf_counts.values, 
        color=[CONFIDENCE_COLORS[c] for c in conf_counts.index],
        edgecolor='black', 
        linewidth=1.5
    )
    ax2.set_ylabel('Number of Pairs', fontsize=11, weight='bold')
    ax2.set_title('Consensus Confidence Levels', fontsize=14, weight='bold', pad=10)
    ax2.set_ylim(0, max(conf_counts.values) * 1.15)
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom', fontsize=11, weight='bold')
    
    # Panel 3: Mutation Type Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    mut_type_counts = df['mut_type'].value_counts()
    bars = ax3.bar(
        range(len(mut_type_counts)), 
        mut_type_counts.values,
        color=['#3498db', '#e67e22'], 
        edgecolor='black', 
        linewidth=1.5
    )
    ax3.set_xticks(range(len(mut_type_counts)))
    ax3.set_xticklabels(['On-Site\n(at p-site)', 'Flanked\n(nearby)'], fontsize=10)
    ax3.set_ylabel('Number of Pairs', fontsize=11, weight='bold')
    ax3.set_title('Mutation Location Type', fontsize=14, weight='bold', pad=10)
    for i, (bar, count) in enumerate(zip(bars, mut_type_counts.values)):
        height = bar.get_height()
        ax3.text(i, height, f'{int(count)}', ha='center', va='bottom', 
                fontsize=11, weight='bold')
    
    # Panel 4: Mutation Type by Tier
    ax4 = fig.add_subplot(gs[0, 3])
    mut_tier_crosstab = pd.crosstab(df['tier'], df['mut_type'])
    mut_tier_crosstab = mut_tier_crosstab.reindex(['Tier 1', 'Tier 2', 'Tier 3'])
    
    x_pos = np.arange(len(mut_tier_crosstab))
    width = 0.35
    
    bars1 = ax4.bar(
        x_pos - width/2, 
        mut_tier_crosstab.get('flanked', 0), 
        width,
        label='Flanked', 
        color='#e67e22', 
        edgecolor='black', 
        linewidth=1
    )
    bars2 = ax4.bar(
        x_pos + width/2, 
        mut_tier_crosstab.get('onsite', 0), 
        width,
        label='On-Site', 
        color='#3498db', 
        edgecolor='black', 
        linewidth=1
    )
    
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(mut_tier_crosstab.index, fontsize=10)
    ax4.set_ylabel('Number of Pairs', fontsize=11, weight='bold')
    ax4.set_title('Mutation Type by Tier', fontsize=14, weight='bold', pad=10)
    ax4.legend(fontsize=9, framealpha=0.9)
    ax4.grid(axis='y', alpha=0.3)
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)
    
    # Panel 5: Top Disease Categories
    ax5 = fig.add_subplot(gs[1, :2])
    
    # Extract and count disease categories
    disease_list = []
    for diseases in df['Disease/Phenotype'].dropna():
        if isinstance(diseases, str):
            for disease in diseases.split('|'):
                disease = disease.strip()
                if disease and disease != 'not provided' and disease != 'not specified':
                    main_disease = disease.split('(')[0].strip()
                    disease_list.append(main_disease)
    
    disease_counts = pd.Series(disease_list).value_counts().head(15)
    
    ax5.barh(
        range(len(disease_counts)), 
        disease_counts.values,
        color='#9b59b6', 
        edgecolor='black', 
        linewidth=1
    )
    ax5.set_yticks(range(len(disease_counts)))
    ax5.set_yticklabels([d[:60] + '...' if len(d) > 60 else d for d in disease_counts.index], 
                        fontsize=9)
    ax5.set_xlabel('Number of Pairs', fontsize=11, weight='bold')
    ax5.set_title('Top 15 Associated Diseases/Phenotypes', fontsize=14, weight='bold', pad=10)
    ax5.grid(axis='x', alpha=0.3)
    
    for i, count in enumerate(disease_counts.values):
        ax5.text(count + 0.5, i, str(count), va='center', fontsize=9, weight='bold')
    
    # Panel 6: Disease Category Breakdown
    ax6 = fig.add_subplot(gs[1, 2:])
    
    def categorize_disease(disease_text):
        if pd.isna(disease_text):
            return 'Unknown'
        disease_text = str(disease_text).lower()
        if 'cancer' in disease_text or 'carcinoma' in disease_text or 'tumor' in disease_text or 'li-fraumeni' in disease_text:
            return 'Cancer'
        elif 'neurodegenerative' in disease_text or 'frontotemporal' in disease_text or 'amyotrophic' in disease_text or 'als' in disease_text or 'parkinson' in disease_text or 'alzheimer' in disease_text or 'dementia' in disease_text or 'pick disease' in disease_text:
            return 'Neurodegeneration'
        elif 'muscular dystrophy' in disease_text or 'myopathy' in disease_text:
            return 'Muscular Disorders'
        elif 'immunodeficiency' in disease_text or 'immune' in disease_text:
            return 'Immune Disorders'
        elif 'metabolic' in disease_text or 'deficiency' in disease_text:
            return 'Metabolic'
        elif 'sleep' in disease_text or 'circadian' in disease_text:
            return 'Sleep/Circadian'
        else:
            return 'Other'
    
    df['disease_category'] = df['Disease/Phenotype'].apply(categorize_disease)
    disease_cat_counts = df['disease_category'].value_counts()
    
    disease_tier_data = []
    for category in disease_cat_counts.index:
        cat_df = df[df['disease_category'] == category]
        tier_dist = cat_df['tier'].value_counts()
        disease_tier_data.append({
            'Category': category,
            'Tier 1': tier_dist.get('Tier 1', 0),
            'Tier 2': tier_dist.get('Tier 2', 0),
            'Tier 3': tier_dist.get('Tier 3', 0),
            'Total': len(cat_df)
        })
    
    disease_tier_df = pd.DataFrame(disease_tier_data).sort_values('Total', ascending=False)
    
    x_pos = range(len(disease_tier_df))
    ax6.bar(x_pos, disease_tier_df['Tier 1'], color=TIER_COLORS['Tier 1'], 
           edgecolor='black', linewidth=0.5, label='Tier 1')
    ax6.bar(x_pos, disease_tier_df['Tier 2'], bottom=disease_tier_df['Tier 1'],
           color=TIER_COLORS['Tier 2'], edgecolor='black', linewidth=0.5, label='Tier 2')
    ax6.bar(x_pos, disease_tier_df['Tier 3'], 
           bottom=disease_tier_df['Tier 1'] + disease_tier_df['Tier 2'],
           color=TIER_COLORS['Tier 3'], edgecolor='black', linewidth=0.5, label='Tier 3')
    
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(disease_tier_df['Category'], rotation=45, ha='right', fontsize=10)
    ax6.set_ylabel('Number of Pairs', fontsize=11, weight='bold')
    ax6.set_title('Disease Categories by Tier', fontsize=14, weight='bold', pad=10)
    ax6.legend(fontsize=9, framealpha=0.9)
    ax6.grid(axis='y', alpha=0.3)
    
    # Panel 7: Top Substrates
    ax7 = fig.add_subplot(gs[2, :])
    substrate_counts = df.groupby('substrate_gene').size().reset_index(name='count')
    substrate_counts = substrate_counts.sort_values('count', ascending=True).tail(12)
    substrate_tier = df.groupby(['substrate_gene', 'tier']).size().unstack(fill_value=0)
    substrate_tier = substrate_tier.reindex(substrate_counts['substrate_gene'])
    
    bars_t1 = ax7.barh(range(len(substrate_tier)), substrate_tier.get('Tier 1', 0), 
                       color=TIER_COLORS['Tier 1'], label='Tier 1', edgecolor='black', linewidth=0.5)
    bars_t2 = ax7.barh(range(len(substrate_tier)), substrate_tier.get('Tier 2', 0),
                       left=substrate_tier.get('Tier 1', 0),
                       color=TIER_COLORS['Tier 2'], label='Tier 2', edgecolor='black', linewidth=0.5)
    bars_t3 = ax7.barh(range(len(substrate_tier)), substrate_tier.get('Tier 3', 0),
                       left=substrate_tier.get('Tier 1', 0) + substrate_tier.get('Tier 2', 0),
                       color=TIER_COLORS['Tier 3'], label='Tier 3', edgecolor='black', linewidth=0.5)
    
    ax7.set_yticks(range(len(substrate_tier)))
    ax7.set_yticklabels(substrate_tier.index, fontsize=11)
    ax7.set_xlabel('Number of Kinase-Substrate Pairs', fontsize=12, weight='bold')
    ax7.set_title('Top 12 Substrates by Kinase Count (Colored by Tier)', fontsize=14, weight='bold', pad=10)
    ax7.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax7.grid(axis='x', alpha=0.3)
    
    plt.suptitle('Kinase-Substrate Pair Analysis: Tiers, Diseases, and Mutation Types', 
                 fontsize=18, weight='bold', y=0.99)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# FIGURE 2: Kinase-Substrate Matrix with Sidebar
# ============================================================================

def generate_viz2(df, output_path='viz2_final_matrix_with_sidebar.png'):
    """
    Generate kinase-substrate matrix with sidebar showing unique pair counts
    
    Matrix: All phosphosite-level interactions (dots colored by tier, sized by confidence)
    Sidebar: Stacked bars showing unique kinase counts per substrate (by tier)
    """
    
    fig = plt.figure(figsize=(28, 16))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 6], wspace=0.05)
    ax_sidebar = fig.add_subplot(gs[0, 0])
    ax_matrix = fig.add_subplot(gs[0, 1])
    
    # Calculate unique pairs
    df['pair_key'] = df['kinase_gene'] + '_' + df['substrate_gene']
    unique_pairs = df.drop_duplicates(subset='pair_key')
    
    # Get ordering
    df_sorted = df.sort_values(['tier', 'substrate_gene', 'kinase_gene'])
    unique_substrates = df_sorted['substrate_gene'].unique()
    unique_kinases = df_sorted['kinase_gene'].unique()
    
    # Create mapping
    substrate_to_idx = {s: i for i, s in enumerate(unique_substrates)}
    kinase_to_idx = {k: i for i, k in enumerate(unique_kinases)}
    
    # Plot matrix points
    for _, row in df.iterrows():
        x = kinase_to_idx[row['kinase_gene']]
        y = substrate_to_idx[row['substrate_gene']]
        color = TIER_COLORS[row['tier']]
        
        if row['consensus_confidence'] == 'HIGH':
            size = 250
            alpha = 0.9
        elif row['consensus_confidence'] == 'MEDIUM':
            size = 150
            alpha = 0.7
        else:
            size = 80
            alpha = 0.5
        
        ax_matrix.scatter(x, y, s=size, c=color, alpha=alpha, 
                         edgecolors='black', linewidth=0.5)
    
    # Set labels for matrix
    ax_matrix.set_xticks(range(len(unique_kinases)))
    ax_matrix.set_xticklabels(unique_kinases, rotation=90, ha='right', fontsize=8)
    ax_matrix.set_yticks(range(len(unique_substrates)))
    ax_matrix.set_yticklabels(unique_substrates, fontsize=9)
    ax_matrix.set_xlabel('Kinase', fontsize=14, weight='bold')
    ax_matrix.set_ylabel('Substrate', fontsize=14, weight='bold')
    ax_matrix.set_title('Kinase-Substrate Interaction Matrix (Each Point = Phosphosite)', 
                 fontsize=16, weight='bold', pad=20)
    ax_matrix.set_axisbelow(True)
    ax_matrix.grid(True, alpha=0.2, linestyle='--')
    
    # Legend for matrix
    legend_elements = [
        mpatches.Patch(facecolor=TIER_COLORS['Tier 1'], edgecolor='black', label='Tier 1 (Validated)'),
        mpatches.Patch(facecolor=TIER_COLORS['Tier 2'], edgecolor='black', label='Tier 2 (Probable)'),
        mpatches.Patch(facecolor=TIER_COLORS['Tier 3'], edgecolor='black', label='Tier 3 (Speculative)'),
    ]
    ax_matrix.legend(handles=legend_elements, loc='upper left', fontsize=11, framealpha=0.9)
    
    # Sidebar: Unique pair counts
    substrate_pair_counts = []
    for substrate in unique_substrates:
        substrate_pairs = unique_pairs[unique_pairs['substrate_gene'] == substrate]
        tier_dist = substrate_pairs['tier'].value_counts()
        substrate_pair_counts.append({
            'substrate': substrate,
            'Tier 1': tier_dist.get('Tier 1', 0),
            'Tier 2': tier_dist.get('Tier 2', 0),
            'Tier 3': tier_dist.get('Tier 3', 0),
            'Total': len(substrate_pairs)
        })
    
    substrate_pair_df = pd.DataFrame(substrate_pair_counts)
    y_positions = np.arange(len(unique_substrates))
    
    ax_sidebar.barh(y_positions, substrate_pair_df['Tier 1'], 
                    color=TIER_COLORS['Tier 1'], edgecolor='black', linewidth=0.5)
    ax_sidebar.barh(y_positions, substrate_pair_df['Tier 2'], 
                    left=substrate_pair_df['Tier 1'],
                    color=TIER_COLORS['Tier 2'], edgecolor='black', linewidth=0.5)
    ax_sidebar.barh(y_positions, substrate_pair_df['Tier 3'], 
                    left=substrate_pair_df['Tier 1'] + substrate_pair_df['Tier 2'],
                    color=TIER_COLORS['Tier 3'], edgecolor='black', linewidth=0.5)
    
    # Add total counts
    for i, row in substrate_pair_df.iterrows():
        total = row['Total']
        ax_sidebar.text(total + 0.3, i, str(int(total)), 
                       va='center', ha='left', fontsize=8, weight='bold')
    
    ax_sidebar.set_yticks(y_positions)
    ax_sidebar.set_yticklabels([])
    ax_sidebar.set_xlabel('# Unique\nKinases', fontsize=10, weight='bold')
    ax_sidebar.set_title('Interaction\nCount', fontsize=12, weight='bold', pad=10)
    ax_sidebar.invert_xaxis()
    ax_sidebar.grid(axis='x', alpha=0.3)
    ax_sidebar.set_axisbelow(True)
    
    # Add note
    note_text = f"Total pairs: {len(unique_pairs)} unique kinase-substrate combinations\nTotal phosphosites: {len(df)} kinase-substrate-site triplets"
    ax_matrix.text(0.02, 0.98, note_text, transform=ax_matrix.transAxes,
                  fontsize=9, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Kinase-Substrate Interaction Matrix with Unique Pair Counts', 
                 fontsize=18, weight='bold', y=0.96)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# FIGURE 4: Pathway Networks with Phosphosite Nodes (OUTWARD FACING)
# ============================================================================

def draw_curved_edge(ax, x0, y0, x1, y1, color, linewidth, alpha, linestyle, curvature=0.2):
    """Draw a curved edge using Bezier curve"""
    dx = x1 - x0
    dy = y1 - y0
    
    # Perpendicular vector
    perp_x = -dy
    perp_y = dx
    
    # Normalize
    length = np.sqrt(perp_x**2 + perp_y**2)
    if length > 0:
        perp_x /= length
        perp_y /= length
    
    # Control point
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    ctrl_x = mid_x + perp_x * curvature
    ctrl_y = mid_y + perp_y * curvature
    
    # Create Bezier path
    verts = [(x0, y0), (ctrl_x, ctrl_y), (x1, y1)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3]
    
    path = Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor='none', edgecolor=color, 
                               linewidth=linewidth, alpha=alpha, 
                               linestyle=linestyle, zorder=1)
    ax.add_patch(patch)

def generate_viz4(df, output_path='viz4_final.png'):
    """
    Generate pathway-specific networks with phosphosite node architecture
    
    Architecture: Kinase (blue, outer) → Phosphosite (red, budding OUTWARD) → Substrate (purple, center)
    - Kinases evenly distributed around perimeter
    - Phosphosites bud OUTWARD from substrates (uniform red color)
    - All p_ind nodes face radially outward (away from center)
    - Curved edges with tier coloring and mutation type styling
    - NO count labels on p_ind nodes
    """
    
    fig, axes = plt.subplots(3, 3, figsize=(28, 26))
    axes = axes.flatten()
    
    # Define pathway groups
    pathways = {
        'Wnt/β-catenin': ['CTNNB1'],
        'p53 Pathway': ['TP53', 'BRCA1'],
        'Tau/Neurodegeneration': ['MAPT', 'TARDBP', 'CCNF'],
        'Cell Cycle/DDR': ['CHEK2', 'CCND2', 'RAF1'],
        'RTK Signaling': ['PTEN', 'ABL1', 'NTRK2', 'FGFR3', 'CBL', 'RET', 'KIT'],
        'Ion Channels': ['SCN5A', 'SLC12A3', 'CFTR'],
        'Metabolism': ['FH', 'PTS'],
        'Circadian': ['PER2'],
        'NF-κB/Immune': ['NFKBIA', 'NFKB2', 'TBK1', 'RPS6KA3', 'NCF1']
    }
    
    for idx, (pathway_name, substrates) in enumerate(pathways.items()):
        if idx >= len(axes):
            break
        
        ax = axes[idx]
        ax.set_aspect('equal')
        
        # Get data for this pathway
        pathway_df = df[df['substrate_gene'].isin(substrates)].copy()
        
        if len(pathway_df) == 0:
            ax.axis('off')
            continue
        
        # Get unique nodes
        substrate_nodes = pathway_df['substrate_gene'].unique()
        kinase_nodes = pathway_df['kinase_gene'].unique()
        pind_nodes = pathway_df.groupby(['substrate_gene', 'p_ind']).first().reset_index()
        
        # Position substrates in center
        if len(substrate_nodes) == 1:
            substrate_pos = {substrate_nodes[0]: (0, 0)}
        else:
            substrate_pos = {}
            for i, sub in enumerate(substrate_nodes):
                angle = 2 * np.pi * i / len(substrate_nodes)
                substrate_pos[sub] = (0.35 * np.cos(angle), 0.35 * np.sin(angle))
        
        # CRITICAL: Position p_ind nodes RADIALLY OUTWARD from substrate
        pind_positions = {}
        pind_info = {}
        pind_to_kinases = {}
        
        for _, pind_row in pind_nodes.iterrows():
            substrate = pind_row['substrate_gene']
            p_ind = pind_row['p_ind']
            
            substrate_pinds = pind_nodes[pind_nodes['substrate_gene'] == substrate]
            n_pinds = len(substrate_pinds)
            pind_list = sorted(substrate_pinds['p_ind'].values)
            pind_idx = pind_list.index(p_ind)
            
            # Calculate angle pointing AWAY FROM CENTER
            substrate_x, substrate_y = substrate_pos[substrate]
            
            # Base angle from origin to substrate (outward direction)
            base_angle = np.arctan2(substrate_y, substrate_x)
            
            # Distribute p_inds around the substrate, biased outward
            if n_pinds == 1:
                # Single p_ind - point directly away from origin
                angle = base_angle
            else:
                # Multiple p_inds - arc centered on outward direction
                arc_width = 2 * np.pi / 3  # 120 degrees
                start_angle = base_angle - arc_width / 2
                angle = start_angle + arc_width * pind_idx / (n_pinds - 1)
            
            substrate_radius = 0.17
            pind_radius = substrate_radius + 0.14  # Short budding distance
            
            pind_x = substrate_x + pind_radius * np.cos(angle)
            pind_y = substrate_y + pind_radius * np.sin(angle)
            
            pind_key = f"{substrate}_p{int(p_ind)}"
            pind_positions[pind_key] = (pind_x, pind_y)
            
            kinases_for_pind = pathway_df[(pathway_df['substrate_gene'] == substrate) & 
                                          (pathway_df['p_ind'] == p_ind)]['kinase_gene'].tolist()
            pind_to_kinases[pind_key] = kinases_for_pind
            
            pind_info[pind_key] = {
                'n_kinases': len(kinases_for_pind),
                'substrate': substrate,
                'p_ind': p_ind,
                'angle': angle
            }
        
        # Position kinases evenly around perimeter
        kinase_positions = {}
        for i, kin in enumerate(kinase_nodes):
            angle = 2 * np.pi * i / len(kinase_nodes)  # Even distribution
            radius = 1.2
            kin_x = radius * np.cos(angle)
            kin_y = radius * np.sin(angle)
            kinase_positions[kin] = (kin_x, kin_y)
        
        # Draw curved edges: kinase -> p_ind
        for _, row in pathway_df.iterrows():
            kinase = row['kinase_gene']
            substrate = row['substrate_gene']
            p_ind = row['p_ind']
            tier = row['tier']
            mut_type = row['mut_type']
            
            pind_key = f"{substrate}_p{int(p_ind)}"
            if pind_key not in pind_positions:
                continue
            
            # Edge appearance
            if tier == 'Tier 1':
                color = TIER_COLORS['Tier 1']
                linewidth = 2.8
                alpha = 0.85
            elif tier == 'Tier 2':
                color = TIER_COLORS['Tier 2']
                linewidth = 2.2
                alpha = 0.75
            else:
                color = '#d3d3d3'
                linewidth = 1.2
                alpha = 0.35
            
            linestyle = '-' if mut_type == 'onsite' else '--'
            
            # Variable curvature for multiple edges to same p_ind
            kinases_to_pind = pind_to_kinases[pind_key]
            n_edges_to_pind = len(kinases_to_pind)
            
            if n_edges_to_pind > 1:
                kinase_idx = kinases_to_pind.index(kinase)
                curvature = -0.4 + 0.8 * kinase_idx / max(1, n_edges_to_pind - 1)
            else:
                curvature = 0.15
            
            kin_x, kin_y = kinase_positions[kinase]
            pind_x, pind_y = pind_positions[pind_key]
            
            draw_curved_edge(ax, kin_x, kin_y, pind_x, pind_y, 
                            color, linewidth, alpha, linestyle, curvature)
        
        # Draw p_ind -> substrate connectors (short, straight)
        for pind_key in pind_positions.keys():
            substrate = pind_key.split('_p')[0]
            pind_x, pind_y = pind_positions[pind_key]
            sub_x, sub_y = substrate_pos[substrate]
            ax.plot([sub_x, pind_x], [sub_y, pind_y], '-',
                    color='#666666', linewidth=2.0, alpha=0.6, zorder=0)
        
        # Draw substrate nodes (purple)
        for substrate in substrate_nodes:
            x, y = substrate_pos[substrate]
            circle = Circle((x, y), 0.17, facecolor=NODE_COLORS['substrate'], 
                          edgecolor='black', linewidth=2.8, alpha=0.95, zorder=10)
            ax.add_patch(circle)
            ax.text(x, y, substrate, ha='center', va='center', 
                   fontsize=10, weight='bold', color='white', zorder=11)
        
        # Draw p_ind nodes (all red, NO count labels)
        for pind_key, (x, y) in pind_positions.items():
            substrate, p_ind_str = pind_key.split('_p')
            p_ind = int(p_ind_str)
            
            n_kinases = pind_info[pind_key]['n_kinases']
            base_size = 0.07
            node_size = base_size + (n_kinases - 1) * 0.01
            
            circle = Circle((x, y), node_size, facecolor=NODE_COLORS['pind'], 
                          edgecolor='black', linewidth=2.0, alpha=0.95, zorder=9)
            ax.add_patch(circle)
            
            # Only p_ind label, NO count
            ax.text(x, y, f'p{p_ind_str}', ha='center', va='center', 
                   fontsize=7, weight='bold', color='white', zorder=11)
            
            # NO count label - count information conveyed by node size only
        
        # Draw kinase nodes (blue, evenly distributed)
        for kinase in kinase_nodes:
            x, y = kinase_positions[kinase]
            
            kinase_data = pathway_df[pathway_df['kinase_gene'] == kinase]
            kinase_tiers = kinase_data['tier'].values
            
            if 'Tier 1' in kinase_tiers or 'Tier 2' in kinase_tiers:
                node_size = 0.09
                edge_width = 2.0
                alpha = 0.95
                font_size = 7
            else:
                node_size = 0.06
                edge_width = 1.0
                alpha = 0.5
                font_size = 5
            
            circle = Circle((x, y), node_size, facecolor=NODE_COLORS['kinase'], 
                          edgecolor='black', linewidth=edge_width, alpha=alpha, zorder=8)
            ax.add_patch(circle)
            
            ax.text(x, y, kinase[:8], ha='center', va='center', 
                   fontsize=font_size, weight='bold', color='white', zorder=11)
        
        # Title and stats
        tier_counts = pathway_df['tier'].value_counts()
        mut_type_counts = pathway_df['mut_type'].value_counts()
        n_unique_pinds = len(pind_nodes)
        
        title_text = f"{pathway_name}"
        stats_text = f"{len(pathway_df)} K-S pairs via {n_unique_pinds} sites | T1={tier_counts.get('Tier 1', 0)} T2={tier_counts.get('Tier 2', 0)} T3={tier_counts.get('Tier 3', 0)}"
        mut_text = f"On-site: {mut_type_counts.get('onsite', 0)} | Flanked: {mut_type_counts.get('flanked', 0)}"
        
        ax.text(0, 1.55, title_text, ha='center', va='center', 
               fontsize=12, weight='bold', transform=ax.transData)
        ax.text(0, 1.38, stats_text, ha='center', va='center', 
               fontsize=8, transform=ax.transData)
        ax.text(0, -1.55, mut_text, ha='center', va='center', 
               fontsize=8, style='italic', color='#555', transform=ax.transData)
        
        ax.set_xlim(-1.75, 1.75)
        ax.set_ylim(-1.75, 1.75)
        ax.axis('off')
    
    # Turn off unused subplots
    for idx in range(len(pathways), len(axes)):
        axes[idx].axis('off')
    
    # Title
    plt.suptitle('Pathway-Specific Phosphorylation Networks\nKinase (Blue) → Phosphosite (Red) → Substrate (Purple) Architecture', 
                 fontsize=18, weight='bold', y=0.98)
    
    # Legend
    legend_elements = [
        Circle((0, 0), 0.05, facecolor=NODE_COLORS['kinase'], edgecolor='black', 
               linewidth=1.5, label='Kinase (evenly distributed)'),
        Circle((0, 0), 0.04, facecolor=NODE_COLORS['pind'], edgecolor='black', 
               linewidth=1.5, label='Phosphosite p_ind (budding outward)'),
        Circle((0, 0), 0.05, facecolor=NODE_COLORS['substrate'], edgecolor='black', 
               linewidth=1.5, label='Substrate (center)'),
        plt.Line2D([0], [0], color=TIER_COLORS['Tier 1'], linewidth=3, 
                   label='Tier 1 (Validated)'),
        plt.Line2D([0], [0], color=TIER_COLORS['Tier 2'], linewidth=2.5, 
                   label='Tier 2 (Probable)'),
        plt.Line2D([0], [0], color='#d3d3d3', linewidth=1.5, alpha=0.4,
                   label='Tier 3 (Speculative)'),
        plt.Line2D([0], [0], color='black', linewidth=2, linestyle='-',
                   label='On-Site (solid)'),
        plt.Line2D([0], [0], color='black', linewidth=2, linestyle='--',
                   label='Flanked (dashed)'),
    ]
    
    fig.legend(handles=legend_elements, loc='lower center', ncol=4, 
              fontsize=11, framealpha=0.95, edgecolor='black', 
              bbox_to_anchor=(0.5, -0.01), title='Network Elements')
    
    note = "Note: All phosphosite nodes face outward from substrate. Node size indicates kinase convergence (larger = more kinases)."
    fig.text(0.5, 0.01, note, ha='center', fontsize=9, style='italic', color='gray')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_path}")
    plt.close()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """
    Main function to generate all three figures
    
    Usage:
        python generate_all_figures_final.py
    
    Customization tips:
        - Modify TIER_COLORS, NODE_COLORS for different color schemes
        - Adjust figsize parameters for different dimensions
        - Change font sizes in individual plot functions
        - Modify pathway groupings in generate_viz4
        - Adjust node sizes and spacing in viz4
    """
    
    print("="*70)
    print("GENERATING ALL VISUALIZATIONS - FINAL VERSION")
    print("="*70)
    
    # Load data
    input_file = '260129_ptmvar_kinase_substrate_pind_type_all.csv'
    print(f"\nLoading data from: {input_file}")
    df = load_data(input_file)
    
    # Generate figures
    print("\nGenerating figures...")
    print("-"*70)
    
    generate_viz1(df)
    generate_viz2(df)
    generate_viz4(df)
    
    print("-"*70)
    print("\n✓ All figures generated successfully!")
    print("\nOutput files:")
    print("  - viz1_final_overview_dashboard.png")
    print("  - viz2_final_matrix_with_sidebar.png")
    print("  - viz4_final.png")
    print("\nFeatures:")
    print("  VIZ1: 7-panel dashboard with disease and mutation statistics")
    print("  VIZ2: Matrix with unique kinase-substrate pair sidebar counts")
    print("  VIZ4: Pathway networks with outward-facing phosphosite nodes")
    print("\n" + "="*70)

if __name__ == '__main__':
    main()
