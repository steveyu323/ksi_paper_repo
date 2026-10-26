"""
Kinase-Substrate Prediction Analysis
Align prediction dataframes, calculate binary classification metrics, and visualize results

Enhanced features:
- Support for arbitrary number of models
- Optional custom model order and colors
- Flexible threshold options (fixed or F1-optimized)
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, matthews_corrcoef,
    confusion_matrix, roc_curve, precision_recall_curve
)
from plotnine import (
    ggplot, aes, geom_bar, geom_line, geom_point, geom_tile,
    facet_wrap, theme_minimal, labs, theme, element_text,
    scale_fill_gradient2, geom_text, coord_flip, position_dodge,
    scale_color_manual, scale_fill_manual, geom_hline
)
import warnings
warnings.filterwarnings('ignore')


def parse_yml_pred(path,return_logits = False):
    eval_test = pd.read_pickle(path)
    if return_logits:
        dat = dat = pd.DataFrame({'kinase_seq': eval_test['kinase_seq'],
                      'substrate_seq':eval_test['substrate_seq'],
                     'probs':eval_test['probs_pos'],
                         'logits':eval_test['logits_pos']})
        return dat
    try:
        dat = pd.DataFrame({'kinase_seq': eval_test['kinase_seq'],
                      'substrate_seq':eval_test['substrate_seq'],
                     'probs':eval_test['probs_pos']})
    except:
        dat = pd.DataFrame({'kinase_seq': eval_test['kinase_seq'],
                      'substrate_seq':eval_test['substrate_seq'],
                     'probs':eval_test['probs']})
    return dat


def generate_color_palette(n_models):
    """
    Generate a visually distinct color palette for n models.
    
    Parameters:
    -----------
    n_models : int
        Number of models/colors needed
    
    Returns:
    --------
    list : List of hex color codes
    """
    # Extended color palette with visually distinct colors
    colors = [
        '#E74C3C',  # Red
        '#3498DB',  # Blue
        '#2ECC71',  # Green
        '#F39C12',  # Orange
        '#9B59B6',  # Purple
        '#1ABC9C',  # Turquoise
        '#E91E63',  # Pink
        '#FF5722',  # Deep Orange
        '#00BCD4',  # Cyan
        '#8BC34A',  # Light Green
        '#FFC107',  # Amber
        '#673AB7',  # Deep Purple
        '#009688',  # Teal
        '#795548',  # Brown
        '#607D8B',  # Blue Grey
        '#FFEB3B',  # Yellow
        '#4CAF50',  # Green
        '#FF9800',  # Orange
        '#9C27B0',  # Purple
        '#03A9F4',  # Light Blue
    ]
    
    if n_models <= len(colors):
        return colors[:n_models]
    else:
        # If we need more colors, cycle through and modify
        extended_colors = []
        for i in range(n_models):
            extended_colors.append(colors[i % len(colors)])
        return extended_colors

    
def output_evaluation_outputs(
    pairs_pred, 
    prediction_dfs,
    model_order=None,
    model_colors=None,
    threshold_mode='optimal',
    fixed_threshold=0.5,
    bar_plot_metrics=None
):
    """
    Evaluate and visualize predictions from multiple models.
    
    Parameters:
    -----------
    pairs_pred : DataFrame
        Reference dataframe with 'kinase_seq', 'substrate_seq', and 'label' columns
    prediction_dfs : dict
        Dictionary mapping model names to prediction dataframes
    model_order : list, optional
        Specify the order of models in plots. If None, uses original dict order.
    model_colors : dict, optional
        Dictionary mapping model names to colors. If None, auto-generates colors.
    threshold_mode : str, default='optimal'
        Threshold strategy for classification:
        - 'optimal': Use F1-optimized threshold for each model
        - 'fixed': Use the same fixed threshold for all models
    fixed_threshold : float, default=0.5
        Threshold value when threshold_mode='fixed'
    bar_plot_metrics : list, optional
        List of metrics to display in the bar plot (p2). If None, defaults to
        ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1', 'MCC'].
        Available metrics: 'Accuracy', 'Precision', 'Recall'/'Sensitivity', 
        'Specificity', 'F1', 'MCC', 'ROC-AUC', 'PR-AUC'
    
    Returns:
    --------
    tuple : (metrics_df, plot_dict) where plot_dict contains all plots
    """
    # ============================================================================
    # STEP 1: Process model order and colors
    # ============================================================================
    print("=" * 80)
    print("KINASE-SUBSTRATE PREDICTION ANALYSIS")
    print("=" * 80)
    
    # Determine model order
    if model_order is None:
        model_order = list(prediction_dfs.keys())
    else:
        # Validate that all models in model_order exist in prediction_dfs
        missing_models = set(model_order) - set(prediction_dfs.keys())
        if missing_models:
            raise ValueError(f"Models specified in model_order not found: {missing_models}")
        
        # Warn if some models in prediction_dfs are not in model_order
        extra_models = set(prediction_dfs.keys()) - set(model_order)
        if extra_models:
            print(f"\nWARNING: Models in prediction_dfs but not in model_order (will be excluded): {extra_models}")
    
    # Generate color palette if not provided
    if model_colors is None:
        colors = generate_color_palette(len(model_order))
        model_colors = dict(zip(model_order, colors))
    else:
        # Validate that all models have colors
        missing_colors = set(model_order) - set(model_colors.keys())
        if missing_colors:
            # Auto-generate colors for missing models
            print(f"\nWARNING: Generating colors for models without specified colors: {missing_colors}")
            additional_colors = generate_color_palette(len(missing_colors))
            for model, color in zip(missing_colors, additional_colors):
                model_colors[model] = color
    
    print(f"\nNumber of models to evaluate: {len(model_order)}")
    print(f"Model order: {model_order}")
    print(f"Threshold mode: {threshold_mode}")
    if threshold_mode == 'fixed':
        print(f"Fixed threshold: {fixed_threshold}")
    
    # ============================================================================
    # STEP 2: Verify and align dataframes
    # ============================================================================
    print("\n[Step 1] Verifying dataframe dimensions...")
    
    # Create a reference from pairs_pred
    pairs_reference = pairs_pred[['kinase_seq', 'substrate_seq', 'label']].copy()
    print(f"Reference pairs_pred shape: {pairs_reference.shape}")

    # Check initial shapes (only for models in model_order)
    print("\nInitial shapes:")
    for name in model_order:
        df = prediction_dfs[name]
        print(f"  {name}: {df.shape}")
    
    # ============================================================================
    # STEP 3: Align predictions with reference pairs
    # ============================================================================
    print("\n[Step 2] Aligning predictions with reference pairs...")
    
    aligned_predictions = {}
    
    for model_name in model_order:
        pred_df = prediction_dfs[model_name]
        # Merge with reference to ensure same order and pairs
        aligned = pairs_reference.merge(
            pred_df[['kinase_seq', 'substrate_seq', 'probs']],
            on=['kinase_seq', 'substrate_seq'],
            how='left'
        )
        
        # Check for missing values
        missing_count = aligned['probs'].isna().sum()
        if missing_count > 0:
            print(f"  WARNING: {model_name} has {missing_count} missing predictions")
            # Option: fill with 0.5 (neutral prediction) or drop
            aligned['probs'] = aligned['probs'].fillna(0.5)
        
        aligned_predictions[model_name] = aligned
        print(f"  {model_name}: aligned shape = {aligned.shape}")
    
    # Verify all have same length
    lengths = [len(df) for df in aligned_predictions.values()]
    assert len(set(lengths)) == 1, "Aligned dataframes have different lengths!"
    print(f"\n✓ All dataframes aligned to length: {lengths[0]}")
    
    # ============================================================================
    # STEP 3.5: Determine thresholds based on mode
    # ============================================================================
    thresholds = {}
    
    if threshold_mode == 'optimal':
        print("\n[Step 2.5] Finding optimal F1-maximizing thresholds...")
        
        for model_name in model_order:
            aligned_df = aligned_predictions[model_name]
            y_true = aligned_df['label'].values
            y_pred_proba = aligned_df['probs'].values
            
            # Calculate precision-recall curve
            precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_proba)
            
            # Calculate F1 scores for each threshold
            # Note: precision_recall_curve returns n+1 precision/recall values but n thresholds
            f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
            
            # Find optimal threshold
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = pr_thresholds[optimal_idx]
            optimal_f1 = f1_scores[optimal_idx]
            
            thresholds[model_name] = optimal_threshold
            
            print(f"  {model_name}:")
            print(f"    Optimal threshold: {optimal_threshold:.4f}")
            print(f"    Max F1 score: {optimal_f1:.4f}")
    
    elif threshold_mode == 'fixed':
        print(f"\n[Step 2.5] Using fixed threshold: {fixed_threshold}")
        for model_name in model_order:
            thresholds[model_name] = fixed_threshold
    
    else:
        raise ValueError(f"Unknown threshold_mode: {threshold_mode}. Use 'optimal' or 'fixed'.")
    
    # ============================================================================
    # STEP 4: Calculate binary classification metrics
    # ============================================================================
    threshold_desc = "F1-Optimized" if threshold_mode == 'optimal' else f"Fixed (t={fixed_threshold})"
    print(f"\n[Step 3] Calculating binary classification metrics with {threshold_desc} thresholds...")
    
    def calculate_metrics(y_true, y_pred_proba, threshold):
        """Calculate comprehensive binary classification metrics"""
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        metrics = {
            'Threshold': threshold,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred, zero_division=0),
            'Recall': recall_score(y_true, y_pred, zero_division=0),
            'F1_Score': f1_score(y_true, y_pred, zero_division=0),
            'ROC_AUC': roc_auc_score(y_true, y_pred_proba),
            'PR_AUC': average_precision_score(y_true, y_pred_proba),
            'MCC': matthews_corrcoef(y_true, y_pred)
        }
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics.update({
            'True_Negative': tn,
            'False_Positive': fp,
            'False_Negative': fn,
            'True_Positive': tp,
            'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'NPV': tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        })
        
        return metrics
    
    # Calculate metrics for each model
    all_metrics = []
    
    for model_name in model_order:
        aligned_df = aligned_predictions[model_name]
        y_true = aligned_df['label'].values
        y_pred_proba = aligned_df['probs'].values
        
        # Use threshold for this model
        threshold = thresholds[model_name]
        metrics = calculate_metrics(y_true, y_pred_proba, threshold)
        metrics['Model'] = model_name
        all_metrics.append(metrics)
        
        print(f"\n{model_name} (threshold={threshold:.4f}):")
        print(f"  Accuracy:  {metrics['Accuracy']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall:    {metrics['Recall']:.4f}")
        print(f"  F1 Score:  {metrics['F1_Score']:.4f}")
        print(f"  ROC-AUC:   {metrics['ROC_AUC']:.4f}")
        print(f"  PR-AUC:    {metrics['PR_AUC']:.4f}")
        print(f"  MCC:       {metrics['MCC']:.4f}")
    
    # Create metrics dataframe
    metrics_df = pd.DataFrame(all_metrics)
    
    # Reorder columns to include Threshold
    metric_cols = ['Threshold', 'Accuracy', 'Precision', 'Recall', 'F1_Score', 
                   'ROC_AUC', 'PR_AUC', 'MCC', 'Specificity', 'NPV']
    metrics_df = metrics_df[['Model'] + metric_cols + 
                            ['True_Positive', 'True_Negative', 
                             'False_Positive', 'False_Negative']]
    
    # Ensure model order is preserved
    metrics_df['Model'] = pd.Categorical(metrics_df['Model'], categories=model_order, ordered=True)
    metrics_df = metrics_df.sort_values('Model')
    
    print("\n" + "=" * 80)
    print(f"SUMMARY METRICS TABLE ({threshold_desc} Thresholds)")
    print("=" * 80)
    print(metrics_df.to_string(index=False))
    
    # ============================================================================
    # STEP 5: Create visualizations with plotnine
    # ============================================================================
    print("\n[Step 4] Creating visualizations...")
    
    # Prepare data for plotting (exclude Threshold from metric comparisons)
    plot_metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1_Score', 
                        'ROC_AUC', 'PR_AUC', 'MCC', 'Specificity', 'NPV']
    
    metrics_long = pd.melt(
        metrics_df,
        id_vars=['Model'],
        value_vars=plot_metric_cols,
        var_name='Metric',
        value_name='Score'
    )
    
    # Ensure model order is preserved in long format
    metrics_long['Model'] = pd.Categorical(metrics_long['Model'], categories=model_order, ordered=True)
    
    # Create color scale for plots
    color_scale = scale_color_manual(values=[model_colors[m] for m in model_order])
    fill_scale = scale_fill_manual(values=[model_colors[m] for m in model_order])
    
    # Plot 1: All metrics comparison
    print("  - Creating comprehensive metrics comparison...")
    p1 = (ggplot(metrics_long, aes(x='Metric', y='Score', fill='Model'))
          + geom_bar(stat='identity', position=position_dodge(width=0.8), width=0.7)
          + theme_minimal()
          + theme(
              axis_text_x=element_text(rotation=45, hjust=1),
              figure_size=(16, 6),
              legend_position='right'
          )
          + labs(
              title=f'All Performance Metrics Comparison ({threshold_desc} Thresholds)',
              x='Metric',
              y='Score'
          )
          + fill_scale
          + geom_hline(yintercept=0.5, linetype='dashed', alpha=0.5)
    )
    
    # Plot 2: Key metrics comparison
    print("  - Creating key metrics comparison...")
    
    # Define default metrics for bar plot if not specified
    if bar_plot_metrics is None:
        bar_plot_metrics = ['Accuracy', 'Precision', 'Sensitivity', 'Specificity', 'F1_Score', 'MCC']
    
    # Validate that requested metrics exist in the metrics dataframe
    available_metrics = plot_metric_cols
    invalid_metrics = [m for m in bar_plot_metrics if m not in available_metrics and m != 'Sensitivity']
    if invalid_metrics:
        print(f"  WARNING: Invalid metrics specified in bar_plot_metrics: {invalid_metrics}")
        print(f"  Available metrics: {available_metrics}")
        # Filter out invalid metrics
        bar_plot_metrics = [m for m in bar_plot_metrics if m in available_metrics or m == 'Sensitivity']
    
    # Handle 'Sensitivity' which is displayed as 'Recall' in the dataframe
    bar_plot_metrics_internal = [m if m != 'Sensitivity' else 'Recall' for m in bar_plot_metrics]
    
    # Filter metrics dataframe to include only requested metrics
    key_metrics = metrics_long[metrics_long['Metric'].isin(bar_plot_metrics_internal)]
    
    # Rename Recall to Sensitivity for clarity
    key_metrics_plot = key_metrics.copy()
    key_metrics_plot['Metric'] = key_metrics_plot['Metric'].replace({'Recall': 'Sensitivity'})
    
    # Ensure metric ordering matches the requested order
    key_metrics_plot['Metric'] = pd.Categorical(
        key_metrics_plot['Metric'], 
        categories=bar_plot_metrics, 
        ordered=True
    )
    
    p2 = (ggplot(key_metrics_plot, aes(x='Metric', y='Score', fill='Model'))
          + geom_bar(stat='identity', position=position_dodge(width=0.8), width=0.7)
          + theme_minimal()
          + theme(
              axis_text_x=element_text(rotation=45, hjust=1),
              figure_size=(14, 6),
              legend_position='right'
          )
          + labs(
              title=f'Key Performance Metrics Comparison ({threshold_desc} Thresholds)',
              x='Metric',
              y='Score'
          )
          + fill_scale
          + geom_hline(yintercept=0.5, linetype='dashed', alpha=0.5)
    )
    
    # Plot 3: Heatmap of metrics
    print("  - Creating heatmap of metrics...")
    heatmap_data = metrics_df.set_index('Model')[plot_metric_cols].T.reset_index()
    heatmap_data = heatmap_data.set_index('index')[model_order].reset_index()  # Ensure model order
    heatmap_long = heatmap_data.melt(id_vars=['index'], var_name='Model', value_name='Score')
    heatmap_long.columns = ['Metric', 'Model', 'Score']
    
    # Ensure model order
    heatmap_long['Model'] = pd.Categorical(heatmap_long['Model'], categories=model_order, ordered=True)
    
    p3 = (ggplot(heatmap_long, aes(x='Model', y='Metric', fill='Score'))
          + geom_tile(color='white', size=0.5)
          + geom_text(aes(label='Score'), format_string='{:.3f}', size=8)
          + scale_fill_gradient2(low='#E74C3C', mid='#F39C12', high='#2ECC71',
                                midpoint=0.5, limits=[0, 1])
          + theme_minimal()
          + theme(
              figure_size=(max(10, len(model_order) * 1.5), 8),
              axis_text_x=element_text(rotation=45, hjust=1)
          )
          + labs(
              title=f'Performance Metrics Heatmap ({threshold_desc} Thresholds)',
              x='Model',
              y='Metric',
              fill='Score'
          )
    )
    
    # Plot 4: ROC Curves
    print("  - Creating ROC curves...")
    roc_data = []
    threshold_points = []
    
    for model_name in model_order:
        aligned_df = aligned_predictions[model_name]
        y_true = aligned_df['label'].values
        y_pred_proba = aligned_df['probs'].values
        fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_proba)
        
        for f, t in zip(fpr, tpr):
            roc_data.append({
                'Model': model_name,
                'FPR': f,
                'TPR': t
            })
        
        # Find FPR and TPR at the chosen threshold
        threshold = thresholds[model_name]
        y_pred_optimal = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred_optimal)
        tn, fp, fn, tp = cm.ravel()
        fpr_optimal = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr_optimal = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        threshold_points.append({
            'Model': model_name,
            'FPR': fpr_optimal,
            'TPR': tpr_optimal
        })
    
    roc_df = pd.DataFrame(roc_data)
    threshold_df = pd.DataFrame(threshold_points)
    
    # Ensure model order
    roc_df['Model'] = pd.Categorical(roc_df['Model'], categories=model_order, ordered=True)
    threshold_df['Model'] = pd.Categorical(threshold_df['Model'], categories=model_order, ordered=True)
    
    p4 = (ggplot(roc_df, aes(x='FPR', y='TPR', color='Model'))
          + geom_line(size=1.2)
          + geom_line(aes(x='FPR', y='FPR'), color='gray', linetype='dashed', 
                      data=pd.DataFrame({'FPR': [0, 1]}), inherit_aes=False)
          + theme_minimal()
          + theme(figure_size=(8, 8), legend_position='bottom')
          + labs(
              title='ROC Curves Comparison',
              x='False Positive Rate',
              y='True Positive Rate'
          )
          + color_scale
    )
    
    # Plot 5: Precision-Recall Curves
    print("  - Creating precision-recall curves...")
    pr_data = []
    pr_threshold_points = []
    
    for model_name in model_order:
        aligned_df = aligned_predictions[model_name]
        y_true = aligned_df['label'].values
        y_pred_proba = aligned_df['probs'].values
        precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
        
        for p, r in zip(precision, recall):
            pr_data.append({
                'Model': model_name,
                'Recall': r,
                'Precision': p
            })
        
        # Calculate precision and recall at the chosen threshold
        threshold = thresholds[model_name]
        y_pred_optimal = (y_pred_proba >= threshold).astype(int)
        precision_optimal = precision_score(y_true, y_pred_optimal, zero_division=0)
        recall_optimal = recall_score(y_true, y_pred_optimal, zero_division=0)
        
        pr_threshold_points.append({
            'Model': model_name,
            'Recall': recall_optimal,
            'Precision': precision_optimal
        })
    
    pr_df = pd.DataFrame(pr_data)
    pr_threshold_df = pd.DataFrame(pr_threshold_points)
    
    # Ensure model order
    pr_df['Model'] = pd.Categorical(pr_df['Model'], categories=model_order, ordered=True)
    pr_threshold_df['Model'] = pd.Categorical(pr_threshold_df['Model'], categories=model_order, ordered=True)
    
    p5 = (ggplot(pr_df, aes(x='Recall', y='Precision', color='Model'))
          + geom_line(size=1.2)
          + theme_minimal()
          + theme(figure_size=(8, 8), legend_position='bottom')
          + labs(
              title='Precision-Recall Curves Comparison',
              x='Recall',
              y='Precision'
          )
          + color_scale
    )
    
    # Plot 6: Model ranking by different metrics
    print("  - Creating model ranking visualization...")
    ranking_data = []
    for metric in plot_metric_cols:
        ranked = metrics_df[['Model', metric]].sort_values(metric, ascending=False).reset_index(drop=True)
        ranked['Rank'] = range(1, len(ranked) + 1)
        ranked['Metric'] = metric
        ranking_data.append(ranked.rename(columns={metric: 'Score'}))
    
    ranking_df = pd.concat(ranking_data, ignore_index=True)
    
    # Ensure model order is preserved
    ranking_df['Model'] = pd.Categorical(ranking_df['Model'], categories=model_order, ordered=True)
    
    p6 = (ggplot(ranking_df, aes(x='Metric', y='Rank', color='Model', group='Model'))
          + geom_line(size=1)
          + geom_point(size=3)
          + theme_minimal()
          + theme(
              axis_text_x=element_text(rotation=45, hjust=1),
              figure_size=(12, 6),
              legend_position='right'
          )
          + labs(
              title='Model Rankings Across Different Metrics',
              x='Metric',
              y='Rank (1 = Best)'
          )
          + color_scale
    )
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE!")
    print("=" * 80)
    print(f"\n✓ All metrics calculated with {threshold_desc} thresholds!")
    print("\nThresholds used:")
    for model_name in model_order:
        print(f"  {model_name}: {thresholds[model_name]:.4f}")
    print("\nUse the following variables to access results:")
    print("  - metrics_df: DataFrame with all metrics")
    print("  - aligned_predictions: Dictionary of aligned prediction dataframes")
    print("  - thresholds: Dictionary of thresholds used")
    print("  - plots: Dictionary containing all plot objects")
    print("  - p1, p2, p3, p4, p5, p6: Individual plot objects")
    
    # Package plots in a dictionary for easier access
    plots = {
        'all_metrics': p1,
        'key_metrics': p2,
        'heatmap': p3,
        'roc_curves': p4,
        'pr_curves': p5,
        'rankings': p6
    }
    
    # Return the metrics dataframe, plots, and other useful objects
    return metrics_df,p2,p4,p5,p6
