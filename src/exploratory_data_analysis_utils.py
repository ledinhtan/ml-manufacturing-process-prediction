import os
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

def compute_corr_matrix(df, input_vars, output_vars=None, method='pearson'):
    """Calculate the Pearson or Spearman correlation matrix"""
    if output_vars:
        corr_matrix = df[input_vars + output_vars].corr(method=method)
        return corr_matrix.loc[input_vars, output_vars]
    else:
        corr_matrix = df[input_vars].corr(method=method)
        return corr_matrix.loc[input_vars]


def rename_corr_matrix(corr_matrix, name_map):
    """Rename the variables"""
    return corr_matrix.rename(index=name_map, columns=name_map)

def plot_scatter(df, x_col, y_col, filename, xlabel, ylabel, hue=None, legend_title=None, palette=None, color=None, dpi=800):       
    """Scatterplot with optional hue and custom legend title"""
    plt.figure(figsize=(10, 8))

    if hue is None:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, s=120, edgecolor='k', color=color)
    else:
        sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue, s=120, edgecolor='k', palette=palette)
        plt.legend(title=legend_title if legend_title else hue, markerscale=2, title_fontsize=20,
                   fontsize=13, frameon=True, framealpha=0.8)

    plt.xlabel(xlabel, fontsize=28, fontweight='bold')
    plt.ylabel(ylabel, fontsize=28, fontweight='bold')
    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    ax = plt.gca()
    ax.tick_params(axis='both', which='major', width=2, length=6, pad=8, labelsize=20)

    plt.tight_layout()
    plt.savefig(f'figures/scatterplots/{filename}.pdf', format='pdf', dpi=dpi)
    plt.savefig(f'figures/scatterplots/{filename}.png', format='png', dpi=dpi)

def plot_interaction_boxplot_pointplot(df, y_col, y_label, box_color='dodgerblue', point_color='deeppink',
                                       filename=None, xlabel='Temperature - Humidity - Volume Flow Rate', dpi=1200):
    """
    boxplot and pointplot grouped by temperature, humidity, and volume flow rate.
    """
    # Group the temperature, humidity and volume flow rate
    if 'interaction_group' not in df.columns:
        median_temp = df['temperature'].median()
        df['temperature_group'] = ['High' if t > median_temp else 'Low' for t in df['temperature']]

        median_humidity = df['humidity'].median()
        df['humidity_group'] = ['High' if h > median_humidity else 'Low' for h in df['humidity']]

        median_flow = df['volume_flow_rate'].median()
        df['flow_group'] = ['High' if f > median_flow else 'Low' for f in df['volume_flow_rate']]

        df['interaction_group'] = df['temperature_group'] + ' - ' + df['humidity_group'] + ' - ' + df['flow_group']

    plt.figure(figsize=(12, 10))

    sns.boxplot(x='interaction_group', y=y_col, data=df, color=box_color)

    sns.pointplot(x='interaction_group', y=y_col, data=df, color=point_color, dodge=True)    

    plt.xlabel(xlabel, fontsize=24, fontweight='bold')
    plt.ylabel(y_label, fontsize=24, fontweight='bold')
    plt.xticks(rotation=45, ha='right', fontsize=21)

    legend_elements = [
        Patch(facecolor=box_color, label='Boxplot'),
        Line2D([0], [0], color=point_color, marker='o', label='Pointplot')
    ]

    plt.legend(handles=legend_elements, fontsize=20, loc='best', frameon=True, framealpha=0.8)
    ax = plt.gca()
    ax.tick_params(axis='both', which='major', width=2, length=6, pad=8, labelsize=20)
    plt.tight_layout()

    if filename:
        plt.savefig(f"figures/boxplot_pointplot/{filename}.pdf", format="pdf", dpi=dpi)
        plt.savefig(f"figures/boxplot_pointplot/{filename}.png", format="png", dpi=dpi)
    plt.show()
