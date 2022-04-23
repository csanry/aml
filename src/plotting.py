from src import config
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from typing import List, Set, Dict, Tuple
from typing import Union, Any, Optional, Iterable, Hashable


class plotviz: 
    def __init__(self, df: pd.DataFrame, 
                 figsize: Tuple = (16, 9), 
                 rows: int = 1, 
                 cols: int = 1, 
                 wspace: float = 0.2, 
                 hspace: float = 0.2, 
                 X: List[str] = None, 
                 y: str = None, 
                 hue: str = None, 
                 hue_order: Dict = None): 
        self.df = df
        self.figsize = figsize
        self.rows = rows
        self.cols = cols
        self.wspace = wspace
        self.hspace = hspace 
        self.X = X
        self.hue = hue
        self.hue_order = hue_order
        self.y = y

    def _make_fig(self): 
        if self.X is None: 
            self.X = []

        # create figure object
        fig = plt.figure(figsize=self.figsize)
        gs = fig.add_gridspec(nrows=self.rows, ncols=self.cols)
        gs.update(wspace=self.wspace, hspace=self.hspace)
        
        run = 0
        for row in range(self.rows):
            for col in range(self.cols):
                globals()[f'ax{run}'] = fig.add_subplot(gs[row, col])
                globals()[f'ax{run}'].set_yticklabels([])
                globals()[f'ax{run}'].tick_params(axis='y', which='both', length=0)
                for s in ['top', 'right', 'left']:
                    globals()[f'ax{run}'].spines[s].set_visible(False)
                run += 1
        return fig, gs
 

    def _check_fig(self): 
        plt.show()
    

    def swarm(self): 
        fig, gs = self._make_fig()

        run = 0
        for col in self.X:
            sns.swarmplot(data=self.df, x=col, y=self.y, hue=self.hue, hue_order=self.hue_order,
                            ax=globals()[f'ax{run}'], dodge=True, palette='pastel')
            globals()[f'ax{run}'].grid(which='major', axis='y', zorder=0, color='black', 
                                        linestyle=config.DEFAULT_PLOT_LINESTYLE, dashes=config.DEFAULT_DASHES)
            globals()[f'ax{run}'].set_title(col.replace('_', ' ').capitalize(), fontsize=config.DEFAULT_AXIS_FONT_SIZE, 
                                            fontweight = 'bold')
            globals()[f'ax{run}'].set_xlabel('')
            globals()[f'ax{run}'].set_ylabel(self.y.replace('_', ' '), fontsize=config.DEFAULT_AXIS_FONT_SIZE)
            run += 1
        plt.show()
    

    def violin(self): 
        fig, gs = self._make_fig()
        
        run = 0
        for col in self.X:
            sns.violinplot(data=self.df, x=col, y=self.y, hue=self.hue, hue_order=self.hue_order,
                        ax=globals()[f'ax{run}'], dodge=True, palette='pastel')
            globals()[f'ax{run}'].grid(which='major', axis='y', zorder=0, color='black', 
                                    linestyle=config.DEFAULT_PLOT_LINESTYLE, dashes=config.DEFAULT_DASHES)
            globals()[f'ax{run}'].set_title(self.X[run].replace('_', ' ').capitalize(), fontsize=12, 
                                        fontweight='bold')
            globals()[f'ax{run}'].set_xlabel('')
            globals()[f'ax{run}'].set_ylabel(self.y.replace('_', ' '), fontsize=config.DEFAULT_AXIS_FONT_SIZE)
            run += 1
        plt.show()


    def box(self): 
        fig, gs = self._make_fig()

        run = 0
        for col in self.X:
            sns.boxplot(data=self.df, x=col, ax=globals()[f'ax{run}'], palette='deep')
            globals()[f'ax{run}'].grid(which='major', axis='x', zorder=0, color='black',
                                      linestyle=config.DEFAULT_PLOT_LINESTYLE, dashes=config.DEFAULT_DASHES)
            globals()[f'ax{run}'].set_title(col.replace('_', ' ').capitalize(), fontsize=config.DEFAULT_AXIS_FONT_SIZE, 
                                           fontweight='bold')
            globals()[f'ax{run}'].set_xlim(self.df[col].min() - 2, self.df[col].max() + 2)
            globals()[f'ax{run}'].set_xlabel('')
            globals()[f'ax{run}'].set_ylabel('')
            globals()[f'ax{run}'].set_xticklabels('')
            run += 1
        plt.show()

            
    def kde(self): 
        fig, gs = self._make_fig()

        run = 0 
        for col in self.X:
            sns.kdeplot(df[col], ax=globals()[f'ax{run}'], shade=True, linewidth=0.5, 
                        color=config.DEFAULT_BLUE)
            globals()[f'ax{run}'].grid(which='major', axis='x', zorder=0, color='black', 
                                      linestyle=config.DEFAULT_PLOT_LINESTYLE, dashes=config.DEFAULT_DASHES)
            globals()[f'ax{run}'].set_title(col.replace('_', ' ').capitalize(), fontsize=config.DEFAULT_AXIS_FONT_SIZE, 
                                           fontweight='bold')
            globals()[f'ax{run}'].set_xlim(self.df[col].min() - 2, self.df[col].max() + 2)
            globals()[f'ax{run}'].set_xlabel('')
            globals()[f'ax{run}'].set_ylabel('')
            run += 1
        plt.show()


    def bar(self):
        fig, gs = self._make_fig()
        
        run = 0 
        for col in self.X:
            df_chart = pd.DataFrame(self.df[col].value_counts() / len(self.df[col]) * 100)
            sns.barplot(x=df_chart.index, y=df_chart[col], ax=globals()[f'ax{run}'], 
                        palette='muted', zorder=3, edgecolor='black', linewidth=0, alpha=0.8)
            globals()[f'ax{run}'].grid(which='major', axis='y', zorder=0, color='black', 
                                      linestyle=':', dashes=(1,5))    
            globals()[f'ax{run}'].set_title(col.replace('_', ' ').capitalize(), fontsize=config.DEFAULT_AXIS_FONT_SIZE, 
                                           fontweight='bold')
            globals()[f'ax{run}'].set_ylabel('%')
            run += 1
        plt.show()


    def barh(self):
        fig, gs = self._make_fig()
        
        run = 0 
        for col in self.X:
            df_chart = pd.DataFrame(self.df[col].value_counts() / len(self.df[col]) * 100)
            sns.barplot(y=df_chart.index, x=df_chart[col], ax=globals()[f'ax{run}'], 
                        palette='muted', zorder=3, edgecolor='black', linewidth=0, alpha=0.8)
            globals()[f'ax{run}'].grid(which='major', axis='x', zorder=0, color='black', 
                                      linestyle=':', dashes=(1,5))    
            globals()[f'ax{run}'].set_title(col.replace('_', ' ').capitalize(), fontsize=config.DEFAULT_AXIS_FONT_SIZE, 
                                           fontweight = 'bold')
            globals()[f'ax{run}'].set_ylabel('')
            globals()[f'ax{run}'].set_xlabel('%')
            run += 1
        plt.show()


def quick_plot(df: pd.DataFrame, hue_var: str = None, diag_kind: str = 'kde') -> None: 
    """Computes a quick summary plot of numeric values 

    Parameters
    ----------
    df : pd.DataFrame :
        pandas DataFrame object

    hue_var : str :
        (Default value = None)

    diag_kind : str :
        (Default value = 'kde')
    
    Returns
    -------
    A pairplot of numeric values 
    """
    sns.pairplot(df, hue=hue_var, diag_kind=diag_kind)
    plt.show() 
    

def set_up_fig(nrows: int = 1, ncols: int = 1, figsize: Tuple = (16, 9)) -> None: 
    """Sets up a fig and axes to plot 

    Parameters
    ----------
    nrows : int :
        (Default value = 1)

    ncols : int :
        (Default value = 1)
    
    figsize : Tuple :
        (Default value = (16, 9))
    
    Returns
    -------
    A figure and array of axes 
    
    """
    fig, ax = plt.subplots(figsize=figsize, nrows=nrows, ncols=nrows)
    for s in ['top', 'right']: 
        ax.spines[s].set_visible(False)
    return fig, ax

def plot_corr(df: pd.DataFrame, rotate_ylabels: bool = False, rotate_xlabels: bool = True) -> None: 
    """Plots the correlations of an input DataFrame 

    Parameters
    ----------
    df : pd.DataFrame :
        pandas DataFrame object

    rotate_ylabels : bool :
        (Default value = False)
    
    rotate_xlabels : bool :
        (Default value = False)
    
    Returns
    -------
    A correlation matrix heatmap 
    
    """
    fig, ax = set_up_fig(nrows=1, ncols=1) 
    cmap = sns.diverging_palette(h_neg=22, h_pos=219, s=80, l=55, as_cmap=True)
    sns.heatmap(df.corr(), ax=ax, vmin=-1, 
                vmax=1, annot=True, square=True, 
                cbar=True, cmap=cmap, fmt='.1g')
    ax.set_title('Correlation plot', fontweight='bold')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=(90 if rotate_ylabels else 0))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=(90 if rotate_xlabels else 0))
    plt.show()



def main() -> None: 
    pass 

if __name__ == '__main__': 
    main() 