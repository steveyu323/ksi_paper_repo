"""Information content of kinase substrate motifs (imported module).

Used by figures/fig06_biological_insights.py; nothing in it reads or writes
fixed paths, so it is reused unchanged apart from the removal of a
demonstration main() that pointed at a scratch directory.

Refined Kinase Substrate Motif Analysis - Information Content Calculator
=========================================================================

Enhanced with:
1. Custom amino acid ordering (D,E,K,R,H,N,Q,S,T,P,G,C,M,A,L,I,V,F,Y,W)
2. Phospho-site column excluded from heatmaps
3. plotnine (ggplot2-style) for visualization
4. Property aggregation: Aliphatic, Hydroxyl, Cyclic, Aromatic, Basic, Acidic

Author: Kinase Substrate Analysis Pipeline
Date: January 2026
"""

import pandas as pd
import numpy as np
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Import plotnine
try:
    from plotnine import *
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    PLOTNINE_AVAILABLE = True
except ImportError:
    PLOTNINE_AVAILABLE = False
    print("WARNING: plotnine not available, falling back to matplotlib")
    import matplotlib.pyplot as plt
    import seaborn as sns

###############################################################################
# CONFIGURATION
###############################################################################

# Custom amino acid order as specified
AA_ORDER = ['D', 'E', 'K', 'R', 'H', 'N', 'Q', 'S', 'T', 'P', 
            'G', 'C', 'M', 'A', 'L', 'I', 'V', 'F', 'Y', 'W']

# Property definitions based on provided classification
AA_TO_PROPERTY = {
    'G': 'Aliphatic',
    'A': 'Aliphatic',
    'V': 'Aliphatic',
    'L': 'Aliphatic',
    'I': 'Aliphatic',
    'S': 'Hydroxyl',
    'C': 'Hydroxyl',
    'T': 'Hydroxyl',
    'M': 'Hydroxyl',
    'P': 'Cyclic',
    'F': 'Aromatic',
    'Y': 'Aromatic',
    'W': 'Aromatic',
    'H': 'Basic',
    'K': 'Basic',
    'R': 'Basic',
    'D': 'Acidic',
    'E': 'Acidic',
    'N': 'Acidic',
    'Q': 'Acidic'
}

# Property order for display
PROPERTY_ORDER = ['Acidic', 'Basic', 'Aromatic', 'Cyclic', 'Hydroxyl', 'Aliphatic']


# Property definitions based on provided classification
# AA_TO_PROPERTY = {
#     # Tiny aliphatic — no or minimal side chain
#     'G': 'Tiny',
#     'A': 'Tiny',
#     # Bulky aliphatic — branched hydrophobic side chains
#     'V': 'Bulky',
#     'L': 'Bulky',
#     'I': 'Bulky',
#     # Sulfur-containing — thioether (Met) and thiol (Cys)
#     'M': 'Sulfur',
#     'C': 'Sulfur',
#     # Aromatic — planar ring systems
#     'F': 'Aromatic',
#     'W': 'Aromatic',
#     'Y': 'Aromatic',
#     # Hydroxyl — polar uncharged, oxygen-bearing side chains
#     'S': 'Hydroxyl',
#     'T': 'Hydroxyl',
#     # Polar — uncharged amide side chains
#     'N': 'Polar',
#     'Q': 'Polar',
#     # Cyclic imino
#     'P': 'Cyclic',
#     # Basic — positively charged at physiological pH
#     'H': 'Basic',
#     'K': 'Basic',
#     'R': 'Basic',
#     # Acidic — negatively charged at physiological pH
#     'D': 'Acidic',
#     'E': 'Acidic',
# }

# Property order for display
# PROPERTY_ORDER = ['Acidic', 'Polar', 'Basic', 'Aromatic', 'Bulky', 'Tiny', 'Cyclic', 'Hydroxyl', 'Sulfur']


# Reverse mapping: property to AAs
PROPERTY_TO_AAS = {}
for aa, prop in AA_TO_PROPERTY.items():
    if prop not in PROPERTY_TO_AAS:
        PROPERTY_TO_AAS[prop] = []
    PROPERTY_TO_AAS[prop].append(aa)

###############################################################################
# CORE FUNCTIONS
###############################################################################

def extract_position_matrix(sequences, phospho_position=7):
    """
    Extract amino acid frequencies at each position.
    Uses custom AA order.
    """
    positions = list(range(-phospho_position, 15-phospho_position))
    
    # Initialize with custom AA order
    freq_matrix = pd.DataFrame(0, index=AA_ORDER, columns=positions)
    
    for seq in sequences:
        if len(seq) != 15:
            continue
        for i, aa in enumerate(seq):
            pos = i - phospho_position
            if aa in AA_ORDER:
                freq_matrix.loc[aa, pos] += 1
    
    return freq_matrix


def extract_property_matrix(sequences, phospho_position=7):
    """
    Extract amino acid PROPERTY frequencies at each position.
    """
    positions = list(range(-phospho_position, 15-phospho_position))
    
    # Initialize with property order
    freq_matrix = pd.DataFrame(0, index=PROPERTY_ORDER, columns=positions)
    
    for seq in sequences:
        if len(seq) != 15:
            continue
        for i, aa in enumerate(seq):
            pos = i - phospho_position
            if aa in AA_TO_PROPERTY:
                prop = AA_TO_PROPERTY[aa]
                freq_matrix.loc[prop, pos] += 1
    
    return freq_matrix


def calculate_background_frequencies(sequences, level='amino_acid'):
    """
    Calculate background frequencies at AA or property level.
    
    Parameters:
    -----------
    sequences : list of str
        Background sequences
    level : str
        'amino_acid' or 'property'
    
    Returns:
    --------
    dict
        Background frequencies
    """
    if level == 'amino_acid':
        aa_counts = Counter()
        for seq in sequences:
            for aa in seq:
                if aa in AA_ORDER:
                    aa_counts[aa] += 1
        
        total = sum(aa_counts.values())
        return {aa: aa_counts[aa] / total for aa in AA_ORDER}
    
    elif level == 'property':
        prop_counts = Counter()
        for seq in sequences:
            for aa in seq:
                if aa in AA_TO_PROPERTY:
                    prop = AA_TO_PROPERTY[aa]
                    prop_counts[prop] += 1
        
        total = sum(prop_counts.values())
        return {prop: prop_counts[prop] / total for prop in PROPERTY_ORDER}


def calculate_information_content(freq_matrix, background_freq=None):
    """
    Calculate information content (bits) at each position.
    
    IC(position) = Σ p(feature|position) × log₂(p(feature|position) / p(feature|background))
    """
    features = freq_matrix.index
    
    # Use uniform background if not provided
    if background_freq is None:
        n_features = len(features)
        background_freq = {f: 1/n_features for f in features}
    
    positions = freq_matrix.columns
    ic_values = []
    
    for pos in positions:
        total = freq_matrix[pos].sum()
        if total == 0:
            ic_values.append(0)
            continue
        
        ic = 0
        for feature in features:
            p_feature = freq_matrix.loc[feature, pos] / total
            if p_feature > 0:
                bg = background_freq.get(feature, 1/len(features))
                ic += p_feature * np.log2(p_feature / bg)
        
        ic_values.append(max(0, ic))
    
    return pd.Series(ic_values, index=positions)


def calculate_ic_contribution_matrix(freq_matrix, background_freq=None):
    """
    Calculate IC contribution of each feature at each position.
    
    Returns matrix with features as rows, positions as columns.
    """
    features = freq_matrix.index
    
    if background_freq is None:
        n_features = len(features)
        background_freq = {f: 1/n_features for f in features}
    
    positions = freq_matrix.columns
    ic_contrib_matrix = pd.DataFrame(0.0, index=features, columns=positions)
    
    for pos in positions:
        total = freq_matrix[pos].sum()
        if total == 0:
            continue
        
        for feature in features:
            p_feature = freq_matrix.loc[feature, pos] / total
            if p_feature > 0:
                bg = background_freq.get(feature, 1/len(features))
                ic_contrib = p_feature * np.log2(p_feature / bg)
                ic_contrib_matrix.loc[feature, pos] = max(0, ic_contrib)
    
    return ic_contrib_matrix


###############################################################################
# VISUALIZATION WITH PLOTNINE
###############################################################################

def plot_ic_heatmap_plotnine(ic_contrib_matrix, title, filename, 
                             exclude_phospho_site=True):
    """
    Create IC contribution heatmap using plotnine (ggplot2 style).
    Falls back to matplotlib if plotnine not available.
    """
    if not PLOTNINE_AVAILABLE:
        # Matplotlib fallback
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        df = ic_contrib_matrix.copy()
        
        # Exclude phospho-site if requested
        if exclude_phospho_site and 0 in df.columns:
            df = df.drop(columns=[0])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create heatmap
        sns.heatmap(df, ax=ax, cmap='YlOrRd',
                   cbar_kws={'label': 'IC Contribution (bits)'},
                   linewidths=0.5, linecolor='white',
                   vmin=0)
        
        # Format position labels
        col_labels = [f"{int(p):+d}" if p != 0 else "0" for p in df.columns]
        ax.set_xticklabels(col_labels, rotation=0, ha='center')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        
        ax.set_xlabel('Position relative to phospho-site', fontsize=12, fontweight='bold')
        ax.set_ylabel('', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {filename}")
        return
    
    # Prepare data
    df = ic_contrib_matrix.copy()
    
    # Exclude phospho-site if requested
    if exclude_phospho_site and 0 in df.columns:
        df = df.drop(columns=[0])
    
    # Convert to long format for plotnine
    df_long = df.reset_index().melt(id_vars='index', 
                                     var_name='Position', 
                                     value_name='IC_contribution')
    df_long.columns = ['Feature', 'Position', 'IC_contribution']
    
    # Convert position to string for proper ordering
    df_long['Position'] = df_long['Position'].astype(str)
    
    # Create ordered position labels
    positions = sorted([int(p) for p in df_long['Position'].unique()])
    position_labels = [f"{p:+d}" if p != 0 else "0" for p in positions]
    df_long['Position'] = pd.Categorical(
        df_long['Position'].astype(int).apply(lambda x: f"{x:+d}" if x != 0 else "0"),
        categories=position_labels,
        ordered=True
    )
    
    # Ensure feature order
    df_long['Feature'] = pd.Categorical(df_long['Feature'], 
                                        categories=list(df.index), 
                                        ordered=True)
    
    # Create plot
    p = (ggplot(df_long, aes(x='Position', y='Feature', fill='IC_contribution'))
         + geom_tile(color='white', size=0.5)
         + scale_fill_gradient(low='#FFFFCC', high='#FF4500', 
                              name='IC\nContribution\n(bits)',
                              limits=[0, df_long['IC_contribution'].max() * 1.1])
         + labs(title=title, x='Position relative to phospho-site', y='')
         + theme_minimal()
         + theme(
             figure_size=(12, 6),
             axis_text_x=element_text(angle=0, ha='center', size=10),
             axis_text_y=element_text(size=9),
             axis_title=element_text(size=12, weight='bold'),
             plot_title=element_text(size=14, weight='bold'),
             panel_grid_major=element_blank(),
             panel_grid_minor=element_blank()
         )+ coord_fixed(ratio=1)
    )
    
    # Save
    p.save(filename, dpi=300, width=12, height=6)
    print(f"  Saved: {filename}")


def plot_comprehensive_plotnine(freq_matrix, ic_contrib_matrix, 
                                information_content, title, filename,
                                exclude_phospho_site=True):
    """
    Create comprehensive 3-panel plot using plotnine.
    
    Panels:
    1. IC contribution heatmap
    2. Frequency heatmap
    3. Total IC bar plot
    """
    if not PLOTNINE_AVAILABLE:
        print("  plotnine not available, using matplotlib fallback...")
        plot_comprehensive_matplotlib(freq_matrix, ic_contrib_matrix, 
                                     information_content, title, filename,
                                     exclude_phospho_site)
        return
    
    # Prepare data
    ic_df = ic_contrib_matrix.copy()
    freq_df = freq_matrix.div(freq_matrix.sum(axis=0), axis=1)
    
    # Exclude phospho-site if requested
    if exclude_phospho_site and 0 in ic_df.columns:
        ic_df = ic_df.drop(columns=[0])
        freq_df = freq_df.drop(columns=[0])
        ic_plot = information_content.drop(0)
    else:
        ic_plot = information_content.copy()
    
    # Panel 1: IC contribution heatmap
    ic_long = ic_df.reset_index().melt(id_vars='index', 
                                       var_name='Position', 
                                       value_name='IC_contribution')
    ic_long.columns = ['Feature', 'Position', 'IC_contribution']
    ic_long['Position'] = pd.Categorical(
        ic_long['Position'].astype(int).apply(lambda x: f"{x:+d}" if x != 0 else "0"),
        categories=[f"{p:+d}" if p != 0 else "0" for p in sorted([int(x) for x in ic_long['Position'].unique()])],
        ordered=True
    )
    ic_long['Feature'] = pd.Categorical(ic_long['Feature'], 
                                       categories=list(ic_df.index), 
                                       ordered=True)
    
    p1 = (ggplot(ic_long, aes(x='Position', y='Feature', fill='IC_contribution'))
          + geom_tile(color='white', size=0.3)
          + scale_fill_gradient(
    low="#FFEDA0",
    high="#0A2A6E",
    name="IC (bits)",
    limits=[0, 0.6],
)
          + labs(title=f'{title} - IC Contribution', x='', y='')
          + theme_minimal()
          + theme(
              figure_size=(12, 4),
              axis_text_x=element_blank(),
              axis_text_y=element_text(size=8),
              plot_title=element_text(size=12, weight='bold'),
              panel_grid=element_blank()
          )+ coord_fixed(ratio=1)
    )
    
    # Panel 2: Frequency heatmap
    freq_long = freq_df.reset_index().melt(id_vars='index',
                                           var_name='Position',
                                           value_name='Frequency')
    freq_long.columns = ['Feature', 'Position', 'Frequency']
    freq_long['Position'] = pd.Categorical(
        freq_long['Position'].astype(int).apply(lambda x: f"{x:+d}" if x != 0 else "0"),
        categories=[f"{p:+d}" if p != 0 else "0" for p in sorted([int(x) for x in freq_long['Position'].unique()])],
        ordered=True
    )
    freq_long['Feature'] = pd.Categorical(freq_long['Feature'],
                                         categories=list(freq_df.index),
                                         ordered=True)
    
    p2 = (ggplot(freq_long, aes(x='Position', y='Feature', fill='Frequency'))
          + geom_tile(color='white', size=0.3)
          + scale_fill_gradient(
    low="#FFEDA0",
    high="#0A2A6E",
    name="IC (bits)",
    limits=[0, 0.6],
)
          + labs(title='Observed Frequencies', x='', y='')
          + theme_minimal()
          + theme(
              figure_size=(12, 4),
              axis_text_x=element_blank(),
              axis_text_y=element_text(size=8),
              plot_title=element_text(size=11, weight='bold'),
              panel_grid=element_blank()
          )+ coord_fixed(ratio=1)
    )
    
    # Panel 3: Total IC bar plot
    ic_bar = pd.DataFrame({
        'Position': [f"{p:+d}" if p != 0 else "0" for p in ic_plot.index],
        'IC': ic_plot.values
    })
    ic_bar['Position'] = pd.Categorical(ic_bar['Position'],
                                       categories=ic_bar['Position'],
                                       ordered=True)
    
    p3 = (ggplot(ic_bar, aes(x='Position', y='IC'))
          + geom_col(fill='steelblue', alpha=0.7, color='black', size=0.3)
          + geom_hline(yintercept=0.5, linetype='dashed', color='gray', alpha=0.5)
          + labs(x='Position relative to phospho-site', y='Total IC (bits)')
          + theme_minimal()
          + theme(
              figure_size=(12, 2),
              axis_text_x=element_text(angle=0, ha='center', size=9),
              axis_text_y=element_text(size=9),
              axis_title=element_text(size=10, weight='bold'),
              panel_grid_major_x=element_blank()
          )
    )
    
    # Save individual plots (plotnine doesn't support subplots easily)
    base_name = filename.replace('.png', '')
    p1.save(f"{base_name}_panel1.png", dpi=300, width=12, height=4)
    p2.save(f"{base_name}_panel2.png", dpi=300, width=12, height=4)
    p3.save(f"{base_name}_panel3.png", dpi=300, width=12, height=2)
    
    print(f"  Saved 3-panel components: {base_name}_panel[1-3].png")


def plot_comprehensive_matplotlib(freq_matrix, ic_contrib_matrix,
                                 information_content, title, filename,
                                 exclude_phospho_site=True):
    """
    Matplotlib fallback for comprehensive plot.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Prepare data
    ic_df = ic_contrib_matrix.copy()
    freq_df = freq_matrix.div(freq_matrix.sum(axis=0), axis=1)
    
    if exclude_phospho_site and 0 in ic_df.columns:
        ic_df = ic_df.drop(columns=[0])
        freq_df = freq_df.drop(columns=[0])
        ic_plot = information_content.drop(0)
    else:
        ic_plot = information_content.copy()
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), 
                            gridspec_kw={'height_ratios': [3, 2, 1]})
    
    # Panel 1: IC contribution
    sns.heatmap(ic_df, ax=axes[0], cmap='YlOrRd',
                cbar_kws={'label': 'IC (bits)'},
                linewidths=0.5, linecolor='white')
    axes[0].set_title(f'{title} - IC Contribution', fontweight='bold', size=12)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('')
    
    # Panel 2: Frequencies
    sns.heatmap(freq_df, ax=axes[1], cmap='RdBu_r', center=0.05,
                cbar_kws={'label': 'Frequency'},
                linewidths=0.5, linecolor='white')
    axes[1].set_title('Observed Frequencies', fontweight='bold', size=11)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('')
    
    # Panel 3: Total IC
    positions = [f"{int(p):+d}" if p != 0 else "0" for p in ic_plot.index]
    axes[2].bar(range(len(positions)), ic_plot.values,
                color='steelblue', alpha=0.7, edgecolor='black')
    axes[2].set_xticks(range(len(positions)))
    axes[2].set_xticklabels(positions)
    axes[2].set_xlabel('Position relative to phospho-site', fontweight='bold')
    axes[2].set_ylabel('Total IC (bits)', fontweight='bold')
    axes[2].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {filename}")


###############################################################################
# MAIN ANALYSIS FUNCTION
###############################################################################

def analyze_kinase_substrates(kinase_name, hit_sequences, background_sequences,
                              level='amino_acid', output_dir='/mnt/user-data/outputs',
                              exclude_phospho_site=True):
    """
    Complete analysis with custom AA order and property grouping.
    
    Parameters:
    -----------
    kinase_name : str
        Name of the kinase
    hit_sequences : list of str
        Hit substrate sequences
    background_sequences : list of str
        Background sequences
    level : str
        'amino_acid' or 'property'
    output_dir : str
        Output directory
    exclude_phospho_site : bool
        Whether to exclude position 0 from heatmaps (default: True)
    """
    print(f"\n{'='*70}")
    print(f"Analyzing {kinase_name} - Level: {level}")
    print(f"{'='*70}")
    print(f"Hit sequences: {len(hit_sequences)}")
    print(f"Background sequences: {len(background_sequences)}")
    print(f"Exclude phospho-site from heatmap: {exclude_phospho_site}")
    
    # Extract frequency matrices
    if level == 'amino_acid':
        print("\nLevel: Individual amino acid")
        print(f"AA order: {' '.join(AA_ORDER)}")
        hit_freq_matrix = extract_position_matrix(hit_sequences)
        bg_freq = calculate_background_frequencies(background_sequences, 'amino_acid')
        feature_label = "Amino Acid"
    
    elif level == 'property':
        print("\nLevel: Amino acid property")
        print(f"Properties: {' > '.join(PROPERTY_ORDER)}")
        print("\nProperty groupings:")
        for prop in PROPERTY_ORDER:
            aas = PROPERTY_TO_AAS[prop]
            print(f"  {prop}: {' '.join(aas)}")
        hit_freq_matrix = extract_property_matrix(hit_sequences)
        bg_freq = calculate_background_frequencies(background_sequences, 'property')
        feature_label = "Property"
    
    else:
        raise ValueError(f"Unknown level: {level}")
    
    # Calculate IC
    print("\nCalculating information content...")
    ic = calculate_information_content(hit_freq_matrix, bg_freq)
    ic_contrib_matrix = calculate_ic_contribution_matrix(hit_freq_matrix, bg_freq)
    
    # Summary statistics
    total_ic = ic.sum()
    
    # Exclude position 0 from total if needed
    if exclude_phospho_site and 0 in ic.index:
        total_ic_no_psite = ic.drop(0).sum()
        max_ic_pos = ic.drop(0).idxmax() if len(ic.drop(0)) > 0 else ic.idxmax()
        max_ic_val = ic.drop(0).max() if len(ic.drop(0)) > 0 else ic.max()
    else:
        total_ic_no_psite = total_ic
        max_ic_pos = ic.idxmax()
        max_ic_val = ic.max()
    
    print(f"\nTotal IC (all positions): {total_ic:.2f} bits")
    if exclude_phospho_site and 0 in ic.index:
        print(f"Total IC (excluding P-site): {total_ic_no_psite:.2f} bits")
    print(f"Max IC: {max_ic_val:.2f} bits at position {max_ic_pos:+d}")
    
    # Identify key positions (excluding phospho-site)
    ic_no_psite = ic.drop(0) if (exclude_phospho_site and 0 in ic.index) else ic
    key_positions = ic_no_psite[ic_no_psite > 0.5].sort_values(ascending=False)
    
    if len(key_positions) > 0:
        print(f"\nKey positions (IC > 0.5 bits, excluding P-site): {len(key_positions)}")
        for pos in key_positions.head(5).index:
            # Get top feature at this position
            total = hit_freq_matrix[pos].sum()
            if total > 0:
                freqs = hit_freq_matrix[pos] / total
                top_feature = freqs.idxmax()
                top_freq = freqs.max()
                print(f"  Position {int(pos):+3d}: {top_feature} "
                      f"({top_freq*100:.1f}%), IC={ic[pos]:.2f} bits")
    else:
        print("\nNo positions with IC > 0.5 bits (excluding P-site)")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # Main heatmap
    # heatmap_file = f"{output_dir}/IC_heatmap_{level}_{kinase_name}.png"
    # plot_ic_heatmap_plotnine(ic_contrib_matrix, 
    #                          f"{kinase_name} - IC Contribution ({feature_label} Level)",
    #                          heatmap_file, exclude_phospho_site)
    
    # Comprehensive plot
    comprehensive_file = f"{output_dir}/IC_comprehensive_{level}_{kinase_name}.png"
    plot_comprehensive_plotnine(hit_freq_matrix, ic_contrib_matrix, ic,
                               kinase_name, comprehensive_file, exclude_phospho_site)
    
    # Save numerical results
    ic_contrib_matrix.to_csv(f"{output_dir}/IC_contribution_matrix_{level}_{kinase_name}.csv")
    print(f"  Saved IC contribution matrix CSV")
    
    summary = pd.DataFrame({
        'position': ic.index,
        'total_IC': ic.values
    })
    summary.to_csv(f"{output_dir}/IC_summary_{level}_{kinase_name}.csv", index=False)
    print(f"  Saved IC summary CSV")
    
    return {
        'information_content': ic,
        'ic_contrib_matrix': ic_contrib_matrix,
        'freq_matrix': hit_freq_matrix,
        'total_ic': total_ic,
        'total_ic_no_psite': total_ic_no_psite
    }

def make_ic_panel1_plot_and_df(
    hit_sequences,
    background_sequences,
    *,
    kinase_name: str = "",
    level: str = "amino_acid",          # "amino_acid" or "property"
    exclude_phospho_site: bool = True,  # drop position 0
    phospho_position: int = 7,          # center index in 15mer
    # Motif-bias blocking at +1 relative to center
    block_plus1: bool = True,
    block_aa_set=frozenset({"P"}),      # AA-level block
    block_prop_set=frozenset({"Cyclic"})# property-level block (must match AA_TO_PROPERTY values)
):
    """
    Returns
    -------
    p1 : plotnine.ggplot
        Panel 1 only: IC-contribution heatmap
    ic_long : pd.DataFrame
        Long-format dataframe used to draw p1 with columns:
        ["Feature", "Position", "IC_contribution"]
    """

    # --- sanitize sequences ---
    hits = pd.Series(hit_sequences).dropna().astype(str).str.strip().tolist()
    bg = pd.Series(background_sequences).dropna().astype(str).str.strip().tolist()

    # Keep only 15-mers (your helpers assume 15)
    hits = [s for s in hits if len(s) == 15]
    bg = [s for s in bg if len(s) == 15]

    # --- +1 bias blocking (applied BEFORE motif computation / visualization) ---
    if block_plus1:
        plus1_idx = phospho_position + 1  # +1 relative to center

        if level == "amino_acid":
            hits = [s for s in hits if s[plus1_idx] not in block_aa_set]
        elif level == "property":
            hits = [s for s in hits if AA_TO_PROPERTY.get(s[plus1_idx], None) not in block_prop_set]
        else:
            raise ValueError("level must be 'amino_acid' or 'property'")

    # --- compute frequency + background + IC contribution matrix ---
    if level == "amino_acid":
        freq_matrix = extract_position_matrix(hits, phospho_position=phospho_position)
        bg_freq = calculate_background_frequencies(bg, level="amino_acid")
    elif level == "property":
        freq_matrix = extract_property_matrix(hits, phospho_position=phospho_position)
        bg_freq = calculate_background_frequencies(bg, level="property")
    else:
        raise ValueError("level must be 'amino_acid' or 'property'")

    ic_contrib = calculate_ic_contribution_matrix(freq_matrix, bg_freq)

    # --- Panel 1 data prep (long format) ---
    ic_df = ic_contrib.copy()
    if exclude_phospho_site and 0 in ic_df.columns:
        ic_df = ic_df.drop(columns=[0])

    ic_long = (
        ic_df.reset_index()
            .melt(id_vars="index", var_name="Position", value_name="IC_contribution")
            .rename(columns={"index": "Feature"})
    )

    # Ordered categorical positions: "-7 ... -1, +1 ... +7" as strings
    pos_ints = sorted([int(x) for x in ic_long["Position"].unique()])
    pos_labels = [f"{p:+d}" if p != 0 else "0" for p in pos_ints]
    ic_long["Position"] = pd.Categorical(
        ic_long["Position"].astype(int).map(lambda x: f"{x:+d}" if x != 0 else "0"),
        categories=pos_labels,
        ordered=True,
    )

    # Keep the row order from the matrix index
    ic_long["Feature"] = pd.Categorical(
        ic_long["Feature"], categories=list(ic_df.index), ordered=True
    )

    # --- Panel 1 plot (IC contribution heatmap) ---
    title = f"{kinase_name} - IC Contribution ({'AA' if level=='amino_acid' else 'AA Property'})".strip(" -")

    p1 = (
        ggplot(ic_long, aes(x="Position", y="Feature", fill="IC_contribution"))
        + geom_tile(color="white", size=0.3)
        + scale_fill_gradientn(
            colors=["#FFEDA0","#0A2A6E"],
            values=[0.0, 1.0],
            name="IC (bits)",
            limits=[0, 0.8],
        )
        # ✅ x-axis label explicitly included for panel 1
        + labs(title=title, x="Position relative to phospho-site", y="")
        + theme_minimal()
        + theme(
            figure_size=(12, 4),
            axis_text_x=element_text(angle=0, ha="center", size=10),
            axis_text_y=element_text(size=8),
            axis_title=element_text(size=11, weight="bold"),
            plot_title=element_text(size=12, weight="bold"),
            panel_grid_major=element_blank(),
            panel_grid_minor=element_blank(),
        )
        + coord_fixed(ratio=1)
    )

    return p1, ic_contrib
