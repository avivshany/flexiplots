
# coding: utf-8

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import logging

logger = logging.getLogger('app')
logger.setLevel(logging.DEBUG)

colormap = ['royalblue', 'tomato', 'orchid', 'mediumseagreen',
            'sandybrown', 'mediumturkuoise', 'khaki', 'lightpink']


class FlexiGrid(object):
    """
    A class for creating a matplotlib subplots grid. Includes different
    methods for visualizing flexible scatterplots and histograms
    """

    def __init__(
        self, df, cols=None, x_vars=None, y_vars=None, show_nans=True,
        fit_reg=True, show_corr=True, hue=None, standardize=False, **kwargs
    ):
        """
        Initializing FlexiGrid class object

        Args:
            df: pandas.DataFrame
            cols: list, optional
                list of column names from df to use in the plot
            {x, y}_vars: list, optional
                list of columns to plot.
                if one is set, than the other must also be set
            show_nans: bool, optional
                if True, shows NaN values in scatter plots
                and their % in histograms
            fit_reg: bool, optional
                if True, draws regression lines in scatterplots
            show_corr: bool, optional
                if True, displays pearson correlation value in scatterplots
            hue: str, optional
                a column name in df to group by to different colors

        Raises:
            TypeError:
                if any of the inputs are not of the expected type
            KeyError:
                if any specified variable is not a column in df
            ValueError:
                if only one of {x, y}_vars is specified but not the other
            IndexError:
                if hue is set but the number of unique values in its column
                is bigger than the number of colors in hue_kws['colors']
        """
        if cols:

            if not isinstance(cols, list):
                raise TypeError('cols must be a list of column names in df')
            else:
                not_cols_in_df_error(df=df, cols=cols)

            if hue:
                cols = cols + [hue]

            self.df = df[cols].copy()

        else:

            if not isinstance(df, pd.DataFrame):
                raise TypeError('df must be a pandas.DataFrame')

            self.df = df.copy()

        self.x_vars = x_vars
        self.y_vars = y_vars
        self.show_nans = show_nans
        self.fit_reg = fit_reg
        self.show_corr = show_corr
        self.hue = hue
        self.kwargs = kwargs

        if self.hue:

            if self.df[self.hue].isnull().any():
                self.df.loc[self.df[self.hue].isnull(), self.hue] = 'NaN'

            self.df[self.hue] = self.df[self.hue].astype('object')
            self.hues_n = self.df[self.hue].value_counts().to_dict()

        # setting default values for visualization aesthetics 
        self.settings_kws = {
            'axis_labels': True, 'ticks_labels': True,
            'despine': True, 'show_grid': True
        }
        self.facets_kws = {
            'width': 4, 'height': 4, 'axes_label_fontsize': 14
        }
        self.scatter_kws = {
            'alpha': 0.5, 'size': None, 'color': 'royalblue'
        }
        self.nan_kws = {
            'facecolor': 'white', 'lower_by_pct': 0.15, 'jitter_frac': 5
        }
        self.line_kws = {
            'width': 2, 'color': 'red', 'extend_by_pct': 0.05
        }
        self.hist_kws = {
            'alpha': 0.7, 'bins': 50, 'color': 'royalblue',
            'show_miss_pct': self.show_nans
        }
        self.hist_text_kws = {
            'fontsize': 12, 'color': 'grey', 'pos': (0.75, 0.9)
        }
        self.grid_kws = {'alpha': 0.3}
        self.corr_kws = {
            'color': 'red', 'alpha': 0.7, 'fontsize': 14, 'pos': (None, 0.95)
        }
        self.hue_kws = {
            'colors': [
                'royalblue', 'tomato', 'mediumseagreen', 'orchid',
                'sandybrown', 'mediumturkuoise', 'khaki', 'lightpink'
            ]
        }

        # checking that inputs are correct, otherwise raising error
        if bool(self.x_vars) != bool(self.y_vars):
            raise ValueError(
                'if one of x_vars, y_vars is set, the other must be also'
            )

        if self.x_vars:
            self.x_vars, self.y_vars = list(x_vars), list(y_vars)
            all_cols = list(set(self.x_vars + self.y_vars))
            not_cols_in_df_error(df=self.df, cols=all_cols)

        if self.hue:
            not_cols_in_df_error(df=self.df, cols=hue)
            self.n_hues = len(self.df[self.hue].unique())

            if self.n_hues > len(self.hue_kws['colors']):
                raise IndexError(
                    """
                    the function supports up to {} different hues, 
                    but the hue column has {} unique values
                    """.format(len(self.hue_kws['colors']), self.n_hues)
                )

        self.pairgrid = False if self.x_vars else True

        if standardize:
            self.df = self.df.apply(z_score)

    def set_settings(self):
        """
        Change values in *_kws settings if they are entered as kwargs.
        See the different settings and their values in flexipairs docstring
        """
        settings_dicts = [
            'settings_kws', 'facets_kws', 'scatter_kws', 'line_kws',
            'hist_kws', 'hist_text_kws', 'corr_kws', 'grid_kws'
        ]

        for handle, kws in self.kwargs.items():

            if handle in settings_dicts:
                setting_obj = getattr(self, handle)

                if isinstance(kws, dict):

                    for name, value in kws.items():

                        if name in setting_obj.keys():
                            setting_obj[name] = value
                        else:
                            logger.warning(f'{name} is not a key in {handle}')

                else:
                    logger.warning(f'{handle} must be a dictionary')

            else:
                logger.warning(f'{handle} is not a pre-defined settings argument')

    def create_grid(self):
        """
        Create a plt.subplots fig and axes objects, based on the number of
        variables used in the grid
        """
        if self.pairgrid:
            colnames = get_df_numeric_cols_list(self.df)
            self.x_vars = colnames
            self.y_vars = colnames

        ncols = len(self.y_vars)
        nrows = len(self.x_vars)
        size = (
            (self.facets_kws['width'] * ncols) + ncols,
            (self.facets_kws['height'] * nrows) + nrows
        )
        self.fig, self.axs = plt.subplots(nrows, ncols, figsize=size)

        return self.fig, self.axs

    def _prettify_ax(self, ax, x_col=None, y_col=None):
        """Set aesthetic settings for each axis, if defined in settings_kws"""
        if self.settings_kws['show_grid']:
            ax.grid(alpha=self.grid_kws['alpha'])

        if self.settings_kws['despine']:
            sns.despine(ax=ax)

        if x_col and y_col:

            if self.settings_kws['axis_labels']:
                size = self.facets_kws['axes_label_fontsize']
                ax.set_xlabel(x_col, fontsize=size);
                ax.set_ylabel(y_col, fontsize=size);

            if not self.settings_kws['ticks_labels']:
                ax.set_xticklabels([])
                ax.set_yticklabels([])

    def hist(self, ax, colname):
        """
        Plot histogram of a pandas series on a given matplotlib axis.
        Optionally draws % of missing values

        Args:
            colname: str
                name of column in self.df to plot
            ax: matplotlib subplots axis to plot on
        """
        nans_exist = self.df[colname].isnull().any()

        if self.hue:
            color_num = 0

            for hue_value in self.df[self.hue].unique():
                cond = self.df[self.hue] == hue_value
                srs = self.df.loc[cond, colname].copy()

                if self.hist_kws['show_miss_pct'] & nans_exist:
                    txt_x = self.hist_text_kws['pos'][0]
                    txt_y_adj = color_num / 12
                    txt_y = (self.hist_text_kws['pos'][1] + 0.05) - txt_y_adj
                    self._print_miss_pct(
                        series=self.df.loc[cond, colname].copy(),
                        color=self.hue_kws['colors'][color_num],
                        pos=(txt_x, txt_y),
                        ax=ax
                    )

                srs = srs.dropna()
                sns.kdeplot(
                    data=srs, color=self.hue_kws['colors'][color_num],
                    shade=True, legend=False, ax=ax,
                    label=f'{hue_value} (n={self.hues_n[hue_value]})'
                )
                color_num += 1
        else:
            srs = self.df[colname].copy()

            if self.hist_kws['show_miss_pct'] & nans_exist:
                self._print_miss_pct(
                    series=self.df[colname],
                    color=self.hist_text_kws['color'],
                    pos=self.hist_text_kws['pos'],
                    ax=ax
                )

            srs = srs.dropna()
            ax.hist(
                srs, bins=self.hist_kws['bins'],
                color=self.hist_kws['color'],
                alpha=self.hist_kws['alpha']
            )

        self._prettify_ax(ax=ax, x_col=colname, y_col=colname)

        return ax

    def _print_miss_pct(self, series, color, pos, ax):
        """Helper function for hist to print out % of NaNs on histogram"""
        n_total = series.size
        n_missing = series.isnull().sum()
        missing_pct = round((n_missing / n_total) * 100)
        ax.annotate(
            s=str(missing_pct) + '% NaNs',
            xycoords=ax.transAxes,
            xy=pos,
            fontsize=self.hist_text_kws['fontsize'],
            color=color
        )

    def _transform_nan(self, df, x_col, y_col):
        """
        Helper function for scatter(). Transform a given df with 2
        columns to be compatible for plotting a scatterplot with NaNs

        Args:
            df: pandas.DataFrame
                data to transform
            {x, y}_col: str
                names of the two columns used for scatterplot

        Returns:
            the inputted pandas.DataFrame with these new columns:
            1.  A bool column named 'NaN' with True for every row with
                a missing value in either x_col or y_col
            2.  For each of {x, y}_col, a new columns with '_fillna' prefix
                where missing values are replaced by the minimum of the
                column minus 30% of the column range. additionally,
                these replaced values are added random uniform noise for
                jitter. the replacement and noise values can be manually
                set using nan_kws

        Raises:
            TypeError: if one of the inputs is not in the expected type
        """
        not_cols_in_df_error(df, [x_col, y_col])
        df = df.copy().reset_index(drop=True)

        if x_col == y_col:
            cols = [x_col, 'NaN']
            df['NaN'] = (df[x_col].isnull())
        else:
            cols = [x_col, y_col, 'NaN']
            df['NaN'] = (df[x_col].isnull()) | (df[y_col].isnull())

        if self.hue:
            cols = cols + [self.hue]

        df = df[cols]

        x_col_fillna = x_col + '_fillna'
        y_col_fillna = y_col + '_fillna'
        rng_x = df[x_col].max() - df[x_col].min()
        rng_y = df[y_col].max() - df[y_col].min()
        lower_x_by = rng_x * self.nan_kws['lower_by_pct']
        lower_y_by = rng_y * self.nan_kws['lower_by_pct']
        df[x_col_fillna] = df[x_col].fillna(df[x_col].min() - lower_x_by)
        df[y_col_fillna] = df[y_col].fillna(df[y_col].min() - lower_y_by)
        jtr_x = lower_x_by / self.nan_kws['jitter_frac']
        jtr_y = lower_y_by / self.nan_kws['jitter_frac']
        jitter_x = pd.Series(
            np.random.uniform(low=-1*jtr_x, high=jtr_x, size=df.shape[0])
        )
        jitter_y = pd.Series(
            np.random.uniform(low=-1*jtr_y, high=jtr_y, size=df.shape[0])
        )
        x_nans = df[x_col].isnull()
        y_nans = df[y_col].isnull()
        df.loc[x_nans, x_col_fillna] = df.loc[x_nans, x_col_fillna] + jitter_x
        df.loc[y_nans, y_col_fillna] = df.loc[y_nans, y_col_fillna] + jitter_y

        return df

    def _regline(self, df, x_col, y_col, ax, color, extend_by=2):
        """
        Draw a regression line on a given matplotlib axis.
        ignores df rows with NaN in any of x_col, y_col
        """
        if x_col != y_col:
            df = df.loc[:, [x_col, y_col]].copy().dropna()
            slope, intercept, _, _, _ = stats.linregress(df[x_col], df[y_col])
            x_range = df[x_col].max() - df[x_col].min()
            line_extend = x_range * self.line_kws['extend_by_pct'] * extend_by
            line_left_x = df[x_col].min() - line_extend
            line_right_x = df[x_col].max() + line_extend
            line_left_y = (slope * line_left_x) + intercept
            line_right_y = (slope * line_right_x) + intercept
            ax.plot(
                [line_left_x, line_right_x], [line_left_y, line_right_y],
                linewidth=self.line_kws['width'], color=color
            );

    def _corr(self, df, x_col, y_col, ax, color, pos):
        """
        Helper function for scatter().
        Calculate and show correlation on a matplotlib axis
        """
        r = df[x_col].corr(df[y_col])
        x_pos = pos[0]
        y_pos = pos[1]

        if not x_pos:
            x_pos = .15 if r > 0 else .8

        text = 'r=' + str(r.round(2))
        size = self.corr_kws['fontsize']
        alpha = self.corr_kws['alpha']
        ax.annotate(
            text, xycoords=ax.transAxes, xy=(x_pos, y_pos),
            fontsize=size, color=color, alpha=alpha
        )

    def _bound_nans(self, df, x_col, y_col, ax, color='gray'):
        """
        Draw lines to mark boundaries between NaNs and
        complete observations in scatter function
        """
        if df[[x_col, y_col]].shape[0] > df[[x_col, y_col]].dropna().shape[0]:
            rng_x = df[x_col].max() - df[x_col].min()
            rng_y = df[y_col].max() - df[y_col].min()
            lower_x_min_by = rng_x * (self.nan_kws['lower_by_pct'] / 2)
            lower_y_min_by = rng_y * (self.nan_kws['lower_by_pct'] / 2)
            nan_line_x = df[x_col].min() - lower_x_min_by
            nan_line_y = df[y_col].min() - lower_y_min_by
            ax.axvline(x=nan_line_x, color=color)
            ax.axhline(y=nan_line_y, color=color)

    def scatter(self, x_col, y_col, ax=None):
        """
        Draw a scatterplot.
        optionally showing NaN values, correlation and regression line

        Args:
            {x, y}_col: columns names in self.df to use for plotting
            
        Returns:
            matplotlib axis with the plot
        
        Helper functions:
            _transform_nan()
            _regline()
            _corr()
            _prettify_ax()
        """
        ax = ax if ax else plt.gca()
        xy_cols = [x_col, y_col]

        if self.show_nans:
            df = self._transform_nan(self.df, x_col, y_col)
            nan_cols = xy_cols + ['NaN', x_col+'_fillna', y_col+'_fillna']
            cols = nan_cols + [self.hue] if self.hue else nan_cols
            df = df[cols]
            df['sctr_x'] = df[x_col + '_fillna']
            df['sctr_y'] = df[y_col + '_fillna']
            self._bound_nans(df=df, x_col=x_col, y_col=y_col, ax=ax)
        else:
            cols = xy_cols + [self.hue] if self.hue else xy_cols
            df = self.df[cols].dropna(subset=xy_cols)
            df['NaN'] = False
            df['sctr_x'] = df[x_col]
            df['sctr_y'] = df[y_col]

        if self.hue:
            color_num = 0

            for hue_value in self.df[self.hue].unique():
                hue_cond = df[self.hue] == hue_value
                sctr_df = df.loc[hue_cond, ['sctr_x', 'sctr_y', 'NaN']]

                map_colors = {
                    True: self.nan_kws['facecolor'],
                    False: self.hue_kws['colors'][color_num]
                }
                ax.scatter(
                    x=sctr_df['sctr_x'],
                    y=sctr_df['sctr_y'],
                    alpha=self.scatter_kws['alpha'],
                    facecolors=sctr_df['NaN'].apply(lambda x: map_colors[x]),
                    edgecolors=self.hue_kws['colors'][color_num],
                    s=self.scatter_kws['size'],
                    label=f'{hue_value} (n={self.hues_n[hue_value]})'
                )

                if self.fit_reg:
                    self._regline(
                        df=df.loc[hue_cond, :], x_col=x_col, y_col=y_col,
                        ax=ax, color=self.hue_kws['colors'][color_num]
                    )

                if self.show_corr:
                    y_pos = self.corr_kws['pos'][1]
                    offset = 0.03
                    x_pos = (1 / (self.n_hues + 1)) * (color_num + 1) - offset
                    self._corr(
                        df=df.loc[hue_cond, :], x_col=x_col, y_col=y_col,
                        ax=ax, color=self.hue_kws['colors'][color_num],
                        pos=(x_pos, y_pos)
                    )

                color_num += 1

        else:
            map_colors = {
                True: self.nan_kws['facecolor'],
                False: self.scatter_kws['color']
            }
            ax.scatter(
                x=df['sctr_x'],
                y=df['sctr_y'],
                alpha=self.scatter_kws['alpha'],
                facecolors=df['NaN'].apply(lambda x: map_colors[x]),
                edgecolors=self.scatter_kws['color'],
                s=self.scatter_kws['size']
            )

            if self.fit_reg:
                self._regline(
                    df=df, x_col=x_col, y_col=y_col, ax=ax,
                    color=self.line_kws['color']
                )

            if self.show_corr:
                self._corr(
                    df=df, x_col=x_col, y_col=y_col, ax=ax,
                    color=self.corr_kws['color'], pos=self.corr_kws['pos']
                )

        self._prettify_ax(x_col=x_col, y_col=y_col, ax=ax)

        return ax


def flexipairs(
    df, cols=None, x_vars=None, y_vars=None, show_nans=True, fit_reg=True,
    show_corr=True, standardize=False, hue=None,
    legend_loc=(1, 0.9), legend_fontsize=14, legend_title_fontsize=16, **kwargs
):
    """
    Plot pairwise relationships in a dataframe.
    Optionally showing missing values, correlations, and regression lines.

    By default, plots a grid of matplotlib subplots axes where each variable
    in df is mapped to a row and a column. Histograms are shown on the
    diagonal of the grid and scatterplots on all other axes.

    It is also possible to select specific variables to be shown separately
    on the grid rows and columns by setting x_vars and y_vars.
    in this case all facets will show scatterplots
    
    The default aesthetics are set in the *_kws dictionaries shown below.
    to adjust these settings enter the name of the *_kws object as an
    argument and assign to it a dictionary with the settings names to change
    as their new values. for example to show the regression lines in black
    instead of red add the following argument when calling the function:
    flexipairs(df, line_kws={'color': 'black'})

    Args:
        df: pandas.DataFrame
            the data to plot
        cols: list, optional
            list of column names from df to use in the plot
        {x, y}_vars. string or list of strings, optional
            names of columns in df to show on each axis.
            if one is set the other must be set also. in this case, the
            grid doesn't have to be a square and no histograms will be shown
        show_nans: bool, optional
            if True, plot missing values in gray on the scatterplots
            and prints the percentage of missing value for each variable
            on its histogram
        fit_reg: bool, optional
            if True, draws regression lines on the scatterplots
        show_corr: bool, optional
            if True, prints pearson correlation coefficients on the scatterplots
        hue: str, optional
            a column name in df to group by to different colors
        standardize: bool, optional
            if True, transform all columns to z-scores
        plt_style:
            matplotlib style to use
        legend_fontsize: int, optional
            font size to use in the legend
        legend_loc: tuple of length 2, optional
            legend location to be used in figlegend bbox_to_anchor

    Returns:
        figure and axes from a matplotlib subplots grid
        
    Raises:
        TypeError: if df is not a pandas.DataFrame
        TypeError: if {x, y}_vars are set and they are not lists of strings
        KeyError: if df has less then 2 columns
        KeyError: if variables specified in {x, y}_vars are not columns in df
        
    Aesthetic settings:
        settings_kws = {'axis_labels': True, 'ticks_labels': True,
                        'despine': True, 'show_grid': True}
        facets_kws = {'width': 4, 'height': 4, 'axes_label_fontsize': 16}
        scatter_kws = {'alpha': 0.5, 'size': None, 'color': 'royalblue'}
        nan_kws = {'color': 'grey', 'lower_by_pct': 0.25, 'jitter_frac': 5}
        line_kws = {'width': 2, 'color': 'red', 'extend_by_pct': 0.05}
        hist_kws = {'alpha': 0.7, 'bins': 50, 'color': 'royalblue',
                    'show_miss_pct': self.show_nans}
        hist_text_kws = {'fontsize': 12, 'color': 'grey', 'pos': (0.7, 0.85)}
        grid_kws = {'alpha': 0.3}
        corr_kws = {'color': 'red', 'alpha' :0.7} 
    """
    # check: df must have at least 2 columns.
    # if *_vars is a string convert it to list
    obj = FlexiGrid(
        df, x_vars=x_vars, y_vars=y_vars, show_nans=show_nans,
        fit_reg=fit_reg, show_corr=show_corr, hue=hue,
        standardize=standardize, **kwargs
    )
    obj.set_settings()
    fig, axs = obj.create_grid()

    for grid_row in range(len(obj.x_vars)):

        for grid_col in range(len(obj.y_vars)):

            x_col = obj.x_vars[grid_row]
            y_col = obj.y_vars[grid_col]

            if obj.pairgrid and grid_row == grid_col:
                curr_ax = axs[grid_row, grid_col]
                obj.hist(ax=curr_ax, colname=x_col)
            else:

                if not isinstance(axs, np.ndarray):
                    # applies when both of x_vars, y_vars are of length 1
                    curr_ax = axs
                elif len(axs.shape) == 1:
                    # applies when only one of x_vars, y_vars is of length 1

                    if len(y_vars) == 1:
                        curr_ax = axs[grid_row]
                    elif len(x_vars) == 1:
                        curr_ax = axs[grid_col]

                else:
                    curr_ax = axs[grid_row, grid_col]

                obj.scatter(x_col=x_col, y_col=y_col, ax=curr_ax)
    if hue:
        _draw_figlegend_from_ax(
            fig=fig, ax=curr_ax,
            loc=legend_loc, title=hue,
            fontsize=legend_fontsize, title_size=legend_title_fontsize
        )

    fig.tight_layout();

    return fig, axs


def _get_layout(n_facets, facet_size, layout, figsize):
    """
    calculate the layout of the array of subplots based on the number of facets.
    allows to set a fixed size for each facet and set the figsize accordingly.
    if layout is predefined, run input validation on the user inputted layout.
    if figsize is predefined, override facet_size dimensions

    Args:
        n_facets: int
            the number of facets to be plotted
        facet_size: 2-tuple
            size in inches of the height and width of each facet
        layout: 2-tuple of ints
            a tuple where the first element specifies the number of rows
            and the second element specifies the number of columns
        figsize: 2-tuple, optional
            size in inches of figure height and width. overrides facet_size

    Returns:
        layout: 2-tuple
            the number of rows and columns in the subplots grid
        figsize: 2-tuple
            the size in inches of figure width and height

    Raises:
        TypeError: if layout is not a 2-tuple of integers
        ValueError: if the specified layout cannot contain all the facets
    """
    if layout:

        if not isinstance(layout, tuple):
            raise TypeError('layout must be a tuple of 2 integers')
        elif len(layout) != 2:
            raise TypeError('layout must be a tuple of 2 integers')
        elif (layout[0] * layout[1]) < n_facets:
            raise ValueError(
                'specified layout dimensions cannot contain all plots')
        elif (layout[0] > n_facets) | (layout[1] > n_facets):
            raise ValueError('specified layout dimensions are too large')

    else:

        root = np.sqrt(n_facets)
        ceil = np.ceil(root)
        floor = np.floor(root)

        if ceil * floor < n_facets:
            floor += 1

        if n_facets == 3:
            rows, cols = (1, 3)
        elif n_facets < 9:
            rows, cols = (floor, ceil)
        else:
            rows, cols = (ceil, floor)

        layout = (int(rows), int(cols))

    if figsize is None:
        figsize = (facet_size[0] * layout[1], facet_size[1] * layout[0])

    return layout, figsize


def _del_empty_facets(fig, axs, rows, cols, n_cols):
    """
    Delete redundant axes created in case that the number of subplots
    in the grid does not fit to a rectangle
    """
    if (rows * cols) > n_cols:
        emptys = (rows * cols) - n_cols

        for j in range(1, emptys+1):
            fig.delaxes(axs[rows-1, cols-j]);

    return fig


def z_score(series):
    """Standardize a series to Z scores"""
    if series.dtype in [float, np.float64, int, np.int64]:
        return (series - series.mean()) / series.std()
    else:
        return series


def obs_in_hist(
    df, id_col, id_values, standardize=False, centrality=None, bins='auto',
    facet_size=(6, 4), plt_style='seaborn', colors=None,
    obs_axvline_kwargs=None, center_axvline_kwargs=None, figlegend_kwargs=None
):
    """
    Plot histograms for each numeric col in df with vertical lines
    representing each selected observation in id_values
    
    Args:
        df: pandas.DataFrame
            the data to plot
        id_col: str
            name of column in df that identifies the observations
        id_values: list (can be str if only one observation is selected)
            values in id_col which represent the observations to plot
        standardize: bool, optional
            if True, transforms all columns to z-scores
        centrality: str, optional
            if set, plots a vertical line with a measure of central tendency.
            must be one of: {'mean', 'median}
        bins: int, optional
            value to assign to the plt.hist bins argument
        facet_size: 2-tuple, optional
             width and height of each facet
        plt_style: str, optional
            matplotlib style to use
        colors: list, optional
            matplotlib color strings to use for the vlines of the observations
        {obs_axvline, center_axvline}_kwargs: dictionaries, optional
            keyword arguments for plt.axvline functions for the observations and
            centrality vertical lines respectively
        figlegend_kwargs: dict, optional
            keyword arguments for plt.figlegend function

    Helper functions:
        _get_layout
        _del_empty_facets
    """
    obs_line_kwargs = dict(alpha=0.8, lw=3, ls='--')
    center_line_kwargs = dict(alpha=0.8, lw=2, ls=':', color='k')
    legend_kwargs = dict(loc='upper right', prop={'size': 12})
    kwargs_mapping = dict(
        obs_line_kwargs=obs_axvline_kwargs,
        center_line_kwargs=center_axvline_kwargs,
        legend_kwargs=figlegend_kwargs
    )

    for preset_kwargs, user_kwargs in kwargs_mapping.items():
        if user_kwargs:
            preset_kwargs.update(user_kwargs)

    if colors is None:
        colors = ['r', 'purple', 'g', 'y', 'sienna', 'orange', 'c']

    # checking that the inputs are correct, raising errors if ID isn't unique
    # and warning if player has NaN's
    df[id_col] = df[id_col].astype('object')
    not_cols_in_df_error(df, cols=[id_col])
    not_value_in_series(series=df[id_col], values=id_values)

    if isinstance(id_values, str):
        id_values = [id_values]
    elif not isinstance(id_values, list):
        raise ValueError(
            'id_values must either be a string or a list of IDs'
        )

    if len(id_values) > 7:
        raise ValueError(
            'currently supports comparison of up to 7 players'
        )

    if standardize:
        df = df.apply(z_score)

    selected_players = df.loc[df[id_col].isin(id_values), :]
    selected_players.set_index(id_col, inplace=True, drop=True)

    if selected_players.shape[0] > len(id_values):
        raise ValueError('A given IDs has multiple instances in the id_col ')

    if selected_players.isnull().all().any():
        logger.warning("""
        all selected players have NaN in at least one variable.
        Histograms for these variable/s are displayed with a gray background
        """)

    # create a dictionary with the player's values for drawing vertical lines
    player_values = selected_players.to_dict(orient='index')

    original_n = df.shape[0]
    df = df.loc[~df[id_col].isin(id_values), :].dropna()
    df.drop(id_col, axis=1, inplace=True)

    if df.shape[0] == 0:
        raise ValueError('at least one variable in df has only missing values')

    if (df.shape[0] * 2) < original_n:
        logger.warning('at least one variable has more than 50% missing values')

    # create subplots
    mpl.style.use(plt_style)
    colnames = get_df_numeric_cols_list(df)
    n_cols = len(colnames)
    layout, figure_size = _get_layout(
        n_facets=n_cols, facet_size=facet_size, layout=None, figsize=None
    )
    fig, axs = plt.subplots(layout[0], layout[1], figsize=figure_size);
    lines = []
    labels = []
    # plotting histograms with vline for current player on subplots
    for col_num in range(n_cols):

        if n_cols == 1:
            ax = axs
        else:
            ax = axs.ravel()[col_num]

        colname = colnames[col_num]
        ax.hist(df[colname], bins=bins);
        ax.set_title(colname, fontsize=16);
        player_num = 0

        for id_, values_dict in player_values.items():
            player_value = player_values[id_][colname]
            curr_line = ax.axvline(
                x=player_value, color=colors[player_num], **obs_line_kwargs
            );
            player_num += 1
            id_ = str(id_)

            if len(id_) < 12:
                curr_label = id_
            else:
                curr_label = id_[:4] + '...' + id_[-4:]

            if curr_label not in labels:
                lines.append(curr_line)
                labels.append(curr_label)

        if centrality:
            center_line = ax.axvline(
                x=df[colname].agg(centrality), **center_line_kwargs
            );
            lines.append(center_line)
            labels.append(centrality)

        if selected_players[colname].isnull().all():
            ax.set_facecolor('gray');

    fig = _del_empty_facets(
        fig=fig, axs=axs, rows=layout[0], cols=layout[1], n_cols=n_cols
    );
    fig.legend(lines, labels, **legend_kwargs);


def _draw_figlegend_from_ax(
    fig, ax, loc, title, fontsize, title_size,
    bbox_loc='upper left'
):
    """
    Draw a figure legend in matplotlib subplots, given an axis.
    Assumes that the axis legend (particularly handles and labels) is the
    same as all other axes in the subplots
    
    Args:
        fig: matplotlib.figure
            figure to draw the legend on
        ax: matplotlib.axis
            axis to take the legend information (handles, labels) from
        loc: tuple of length 2, optional
            legend location to be used in figlegend bbox_to_anchor
        title: str
            legend title
        fontsize: int
            legend text fontsize
        title_size: int
            legend title fontsize

    Returns:
        matplotlib.figure object
    """
    handles, labels = ax.get_legend_handles_labels();

    if 'nan' in labels:
        nan_loc = labels.index('nan')
        del labels[nan_loc]
        del handles[nan_loc]

    fig.legend(
        handles=handles, labels=labels, loc=bbox_loc, bbox_to_anchor=loc,
        title=title, fontsize=fontsize, title_fontsize=title_size
    );

    return fig;


def _set_layout(layout, n_facets):
    """
    Input validation for manually setting matplotlib.subplots grid shape.

    Args:
        layout: tuple of 2 ints
            a tuple where the first element specifies the number of rows
            and the second element specifies the number of columns
        n_facets: int
            the number of facets to be plotted

    Returns:
        A tuple specifying the number of rows and columns in the grid

    Raises:
        TypeError: if shape is not a tuple of two integers
        ValueError: if the specified shape cannot contain all the facets
    """
    if not isinstance(layout, tuple):
        raise TypeError('layout must be a tuple of 2 integers')
    elif len(layout) != 2:
        raise TypeError('layout must be a tuple of 2 integers')
    elif (layout[0] * layout[1]) < n_facets:
        raise ValueError('specified layout dimensions cannot contain all plots')
    elif (layout[0] > n_facets) | (layout[1] > n_facets):
        raise ValueError('specified layout dimensions are too large')

    return layout


# class AdjustGrid(object):


#     def __init__(self):
#         """"""
#         pass


#     def set_layout(self, shape, n_facets):
#         """
#         Input validation for manually setting matplotlib.subplots grid shape.

#         Args:
#             shape: tuple of 2 ints
#                 a tuple where the first element specifies the number of rows
#                 and the second element specifies the number of columns
#             n_facets: int
#                 the number of facets to be plotted

#         Returns:
#             A tuple specifying the number of rows and columns in the grid

#         Raises:
#             TypeError: if shape is not a tuple of two integers
#             ValueError: if the specified shape cannot contain all the facets
#         """
#         if not isinstance(shape, tuple):
#             raise TypeError('shape must be a tuple of 2 integers')
#         elif len(shape) != 2:
#             raise TypeError('shape must be a tuple of 2 integers')
#         elif (shape[0] * shape[1]) < n_facets:
#             txt = """The specified dimensions of shape are smaller than
#             the number of numeric columns"""
#             raise ValueError(txt)

#         return shape


#     def get_layout(self, n_facets):
#         """
#         Calculate the shape of the array of subplots based on how many
#         subplots there are. allows to set a fixed size for each subplot
#         and change the figure size accordingly
#         """
#         root = math.sqrt(n_facets)
#         rows = math.ceil(root)
#         cols = math.floor(root)

#         if cols * rows < n_facets:
#             cols = math.ceil(root)

#         return rows, cols


#     def del_empty_facets(self, fig, axs, rows, cols, n_facets):
#         """
#         If the number of subplots in the grid does not fit to a rectangle,
#         delete the extra axes that were created
#         """
#         if (rows * cols) > n_facets:
#             emptys = (rows * cols) - n_facets

#             for j in range(1, emptys + 1):
#                 fig.delaxes(axs[rows - 1, cols - j]);

#         return fig


#     def _draw_figlegend_from_ax(fig, ax, loc, title, fontsize):
#         """
#         Draw a figure legend in matplotlib subplots, given an axis.
#         Assumes that the axis legend (particularly handles and labels)
#         is the same as all other axes in the subplots

#         Args:
#             fig: matplotlib.figure
#                 figure to draw the legend on
#             ax: matplotlib.axis
#                 axis to take the legend information (handles, labels) from
#             loc: tuple of length 2, optional
#                 legend location to be used in figlegend bbox_to_anchor
#             title: str
#                 legend title
#             fontsize: int
#                 legend text fontsize

#         Returns:
#             matplotlib.figure object
#         """
#         handles, labels = ax.get_legend_handles_labels();

#         if 'nan' in labels:
#             nan_loc = labels.index('nan')
#             del labels[nan_loc]
#             del handles[nan_loc]

#         fig.legend(handles=handles, labels=labels,
#                    bbox_to_anchor=loc, title=title, fontsize=fontsize);

#         return fig;


def univariate(
    df, cols=None, hue=None, layout=None, centrality=False,
    figsize=None, facet_size=(4.2S, 4), suptitle=None, legend_loc=(1, 1),
    hue_colors=None, **kwargs
):
    """
    Plot histograms for all numeric columns in a dataframe.
    Optionally set the number of rows/columns in the plot matrix

    Args:
        df: pandas.DataFrame
        cols: list or str, optional
            column names in df to plot
        hue: str, optional
            column name to plot separate histograms in different colors by
        layout: tuple of length 2, optional
            number of facets rows (1st element) and columns (2nd element)
            in the plot matrix. the multiplication of the two elements
            must fit the number of numeric columns in the dataframe
        centrality: str, optional
            if set, plots a vertical line with a measure of central tendency.
            must be one of: {'mean', 'median}
        figsize: 2 tuple, optional
            size in inches of figure width and height. overrides facet_size
        facet_size: 2 tuple, optional
            size in inches of the width and height of each facet respectively
        suptitle: str, optional
            figure title
        legend_loc: tuple of length 2, optional
            legend location to be used in figlegend bbox_to_anchor
        hue_colors: list, optional
            list of color names to use for the different hues

    Returns:
        figure and axes from a matplotlib subplots grid

    Helper functions:
        not_cols_in_df_error
        get_df_numeric_cols_list
        _get_layout
        _del_empty_facets
        _draw_figlegend_from_ax

    Raises:
        Errors for input validation (coming from helper functions)
    """
    # setting default kws and updating them based on user input
    settings = {
        'distplot_kws': {
            'hist': False if hue else True,
            'kde': True,
            'axlabel': False,
            'kde_kws': {
                'lw': 2,
                'legend': True if centrality else False,
            },
            'hist_kws': {}
        },
        'axvline_kws': {'ls': '--', 'lw': 1},
        'legend_kws': {'title_size': 14, 'fontsize': 12},
        'ax_title_kws': {'fontsize': 14},
        'tick_params_kws': {
            'axis': 'both', 'which': 'major', 'labelsize': 8
        },
        'suptitle_kws': {'t': suptitle}
    }
    settings['distplot_kws']['hist_kws'].update(
        {'alpha': 0.5 if settings['distplot_kws']['kde'] else 1}
    )
    one_level_kws = [
        'legend_kws', 'ax_title_kws', 'tick_params_kws', 'suptitle_kws'
    ]

    if kwargs:

        for kwargs_k, kwargs_v in kwargs.items():

            if kwargs_k in one_level_kws:
                settings[kwargs_k].update(kwargs_v)
            elif kwargs_k == 'distplot_kws':

                if isinstance(kwargs_v, dict):

                    for level2_k, level2_v in kwargs_v.items():

                        if isinstance(level2_v, dict):
                            settings[kwargs_k][level2_k].update(level2_v)
                        else:
                            settings[kwargs_k][level2_k] = level2_v

                else:
                    settings[kwargs_k].update(kwargs[kwargs_k])
            else:
                logger.warning(kwargs_k + ' is unknown')

    # input validation and adjustment
    if cols:

        if isinstance(cols, str):
            cols = [cols]

        not_cols_in_df_error(df=df, cols=cols)
        original_columns = cols
        numeric_cols = get_df_numeric_cols_list(df.loc[:, cols])
    else:
        original_columns = df.columns.tolist()
        numeric_cols = get_df_numeric_cols_list(df)

    cols_to_use = [col for col in numeric_cols if col != hue]

    if hue:
        not_cols_in_df_error(df=df, cols=hue)
        cols_to_use = cols_to_use + [hue]

        if hue_colors is None:
            hue_colors = colormap

    if len(cols_to_use) < len(original_columns):
        non_numeric_cols = [
            col for col in original_columns if col not in numeric_cols
        ]

        if hue:
            non_numeric_cols = [
                col for col in non_numeric_cols if col != hue
            ]

        if len(non_numeric_cols) > 0:
            txt = 'the following columns are not numeric and cannot be plotted '
            logger.warning(txt + str(non_numeric_cols))

    # plot data and adjust aesthetics
    df = df.copy().loc[:, cols_to_use]
    layout, figure_size = _get_layout(
        n_facets=len(numeric_cols), facet_size=facet_size,
        layout=layout, figsize=figsize
    )
    fig, axs = plt.subplots(layout[0], layout[1], figsize=figure_size);

    for col_num in range(len(numeric_cols)):

        if len([col for col in cols_to_use if col != hue]) == 1:
            ax = axs;
        else:
            ax = axs.ravel()[col_num];

        colname = numeric_cols[col_num]

        if hue:

            for color_num, hue_value in enumerate(df[hue].unique()):
                series = df.loc[df[hue] == hue_value, colname].dropna()
                curr_color = hue_colors[color_num]

                if centrality:
                    center = series.agg(centrality)
                    ax.axvline(
                        x=center, color=curr_color,
                        **settings['axvline_kws']
                    );
                    curr_label = f'm({hue_value}) = {center:.2f}'
                else:
                    curr_label = str(hue_value)

                sns.distplot(
                    series, color=curr_color, label=curr_label,
                    ax=ax, **settings['distplot_kws']
                );
        else:
            series = df[colname].dropna()

            if centrality:
                center = series.agg(centrality)
                ax.axvline(x=center, **settings['axvline_kws']);
                curr_label = f'm = {center:.2f}'
                settings['distplot_kws']['kde_kws']['label'] = curr_label

            sns.distplot(series, ax=ax, **settings['distplot_kws'])

        ax.set_title(colname, **settings['ax_title_kws']);
        ax.tick_params(**settings['tick_params_kws'])

    if bool(hue) & (not centrality):
        _draw_figlegend_from_ax(
            fig=fig, ax=ax, loc=legend_loc, title=hue,
            **settings['legend_kws']
        )

    if suptitle:
        fig.suptitle(**settings['suptitle_kws'])

    fig.tight_layout();
    fig = _del_empty_facets(
        fig=fig, axs=axs, rows=layout[0], cols=layout[1], n_cols=len(numeric_cols)
    )

    return fig, axs;


def compare_groups_same_scale(
    df, dvs, hue, kind='boxplot', figsize=(8, 6), order=None, colors=None,
    color_by=None, colors_lengths=None, vars_to_float=False, **kwargs
):
    """
    Plot differences between groups on one axis with multiple plots.
    Plot types can be one of: 'boxplot', 'violin', 'bar'.

    Args:
        df: pandas.DataFrame containing all data
        dvs: list
            dependent variables to plot on the y axis
        hue: str
            categorical variable to group by (used in the hue arg of the plot).
            used for coloring the groups, if color_by is not set
        kind: str, optional
            seaborn plot to use. supports: 'boxplot', 'violinplot' 'barplot'
        figsize: tuple of length 2, optional
            height and width of the plot in inches
        color_by: str, optional
            categorical variable to color by in case color should be
            set separately from the grouping variable.
        colors_lengths: list (tuple also works), optional
            the number of instances for each color group in color_by.
            e.g to have 3 bars of one color and 4 bars of another color
            set color_lengths to [3, 4]. the sum of colors_lengths must be
            equal to the number of different categories in hue
        order: list, optional
            order to plot the categorical levels in
        colors: list, optional
            colors to use in case color_by is not None.
            defaults to colormap
        vars_to_float: bool, optional
            if True, all dependent variables' dtype is changed to float

    Returns:
        Matplotlib axis
    """
    if colors is None:
        colors = colormap

    kws = {
        'legend': {'loc': 'upper right', 'fontsize': 8},
        'barplot': {'ci': 95},
        'boxplot': {'showfliers': True},
        'violinplot': {'split': True, 'inner': 'box', 'cut': 0},
        'tick_params': {'axis': 'both', 'which': 'major', 'labelsize': 8}
    }

    if kwargs:

        for kwargs_k, kwargs_v in kwargs.items():

            if kwargs_k.split('_kws')[0] in kws.keys():
                kws[kwargs_k.split('_kws')[0]].update(kwargs_v)
            else:
                raise KeyError(kwargs_k + ' is unknown')

    if bool(color_by) != bool(colors_lengths):
        raise ValueError(
            'if color_by or colors_lengths is defined, the other must also be'
        )

    if color_by:
        palette = []
        df = df.sort_values(by=color_by)

        if len(colors_lengths) > len(colors):
            raise ValueError(f'currently supports up to {len(colors)} different colors')

        for i in range(len(colors_lengths)):
            new_colors = [colors[i]] * colors_lengths[i]
            palette = palette + new_colors

        kws[kind]['palette'] = palette
        id_vars = [color_by, hue]
    else:
        id_vars = [hue]

    if vars_to_float:

        for col in dvs:
            df[col] = df[col].astype(float)

    n_hues = len(df[hue].unique())

    if kind == 'violinplot':

        if (n_hues != 2) & kws[kind]['split']:
            logger.warning(
                """Seaborn.violinplot requires exactly 2 hue levels for split.
                Resorting to split=False"""
            )
            kws[kind]['split'] = False

    cols = dvs + id_vars
    df = df.loc[:, cols].melt(id_vars=id_vars).dropna()
    plot = getattr(sns, kind);
    ax = plot(
        x='variable', y='value', hue=hue, data=df,
        order=order, **kws[kind]
    );

    ax.legend(**kws['legend']);
    ax.tick_params(**kws['tick_params'])

    return ax;


def compare_groups_diff_scales(
    df, x, y_cols=None, hue=None, kind='boxplot', layout=None, facet_size=(4, 4),
    figsize=None, barplot_kws=None, boxplot_kws=None, violinplot_kws=None,
    figlegend_kws=None, tick_params_kws=None, set_title_kws=None
):
    # if one var is bool force for same_scale = False
    """
    Plot differences between groups on separate subplots.
    Numeric dtypes will be presented on boxplots and booleans on barplots
    (as proportion of True). all other dtypes will not be plotted

    Args:
        df: pandas.DataFrame containing all data
        x: str
            df column name of independent variable to appear on the x axis.
            by default uses this column in the hue argument  of the seaborn
            plotting function to color by its levels, unless hue is set
        y_cols: list, optional
            df column names of dependent variables to plot on different facets
            if not set, uses all columns in df except for the columns set to x
        hue: str, optional
            df column name to color by its levels. use in case the hue to color
            by should be different than the column defined for the x argument
        kind: str, optional
            seaborn plot to use. supports: 'boxplot', 'violinplot' 'barplot'
        layout: tuple of length 2, optional
            number of facets rows (1st element) and columns (2nd element)
            in the plot matrix. the multiplication of the two elements
            must fit the number of numeric columns in the dataframe
        facet_size: 2-tuple
            width and height of each subplot in inches
        figsize: tuple, optional
            size in inches of the figure height and weight respectively.
            if set, overrides the facet_size argument
        {boxplot, violinplot, barplot}_kws: dictionaries, optional
            Keyword arguments for underlying seaborn plotting functions
        {figlegend, tick_params, set_title}_kws: dictionaries, optional
            Keyword arguments for underlying matplotlib axis methods

    Returns:
        Matplotlib figure and axes

    Raises:

    """
    if y_cols is None:
        y_cols = df.columns.tolist()

    if not hue:
        hue = x

    # input validation
    for str_input in [x, hue]:

        if not isinstance(str_input, str):
            raise TypeError('the {str_input} argument requires a string input')

    if not isinstance(y_cols, list):
        raise TypeError('the y_cols argument requires a list input')

    if not isinstance(df, pd.DataFrame):
        raise TypeError('the df argument requires a pandas.DataFrame input')

    for col in y_cols + [x, hue]:

        if col not in df.columns:
            raise KeyError(col + ' is not a column in the dataframe')

    # set plot aesthetics
    y_cols = [col for col in y_cols if col not in [x, hue]]
    n_cols = len(y_cols)
    barplot_kwargs = dict(ci=95, dodge=False)
    boxplot_kwargs = dict(showfliers=True, dodge=False)
    violinplot_kwargs = dict(split=True, inner='stick', cut=0, dodge=False)
    figlegend_kwargs = dict(bbox_to_anchor=[1, 1], loc='upper left', fontsize=10)
    tick_params_kwargs = dict(labelsize=10, labelrotation=0)
    set_title_kwargs = dict(fontsize=13)

    if (len(df[x].unique()) != 2) & (kind == 'violinplot'):
        violinplot_kwargs.update(dict(split=False, inner='box'))

    default_kws = [
        barplot_kwargs, boxplot_kwargs, violinplot_kwargs,
        figlegend_kwargs, tick_params_kwargs, set_title_kwargs
    ]
    user_kws = [
        barplot_kws, boxplot_kws, violinplot_kws,
        figlegend_kws, tick_params_kws, set_title_kws
    ]

    for i, kwargs_dict in enumerate(default_kws):

        if user_kws[i]:
            kwargs_dict.update(user_kws[i])

    layout, figure_size = _get_layout(
        n_facets=n_cols, facet_size=facet_size,
        layout=layout, figsize=figsize
    )
    fig, axs = plt.subplots(layout[0], layout[1], figsize=figure_size);
    df = df.copy().sort_values(by=hue)
    non_numeric_cols = []

    for col_num in range(n_cols):
        colname = y_cols[col_num]

        if n_cols == 1:
            ax = axs
        else:
            ax = axs.ravel()[col_num]

        if df[colname].dtype == 'bool':
            curr_kind = 'barplot'
        elif pd.api.types.is_numeric_dtype(df[colname].dtype):
            curr_kind = kind
        else:
            non_numeric_cols.append(colname)

        plot = getattr(sns, curr_kind);
        kinds_kwargs = {
            'barplot': barplot_kwargs, 'boxplot': boxplot_kwargs,
            'violinplot': violinplot_kwargs
        }
        plot_kws = kinds_kwargs[curr_kind]
        curr_plot = plot(y=colname, x=x, hue=hue, data=df, ax=ax, **plot_kws);
        ax.set_title(colname, **set_title_kwargs)
        ax.set_ylabel('')
        ax.legend_.remove()
        ax.tick_params(**tick_params_kwargs)
        ax.set_xlabel('')

    fig = _del_empty_facets(
        fig=fig, axs=axs, rows=layout[0], cols=layout[1], n_cols=n_cols
    );
    handles, labels = curr_plot.get_legend_handles_labels();
    fig.legend(handles=handles, labels=labels, title=hue, **figlegend_kwargs);
    fig.tight_layout();

    if non_numeric_cols:
        warn = ' columns are not numeric or bool and cannot be plotted'
        logger.warning(str(non_numeric_cols) + warn)

    return fig, axs;


def not_cols_in_df_error(df, cols):
    """
    Raises error if df is not a pandas.DataFrame
    or if one of cols is not a column in df

    Args:
        df: pandas.DataFrame
        cols: str or list of strings
            name of the column/s to inspect

    Raises:
        TypeError: if one of the inputs is not in the expected type
        KeyError: if any of cols is not the name of a column in df
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError('df input must be a pandas.DataFrame')

    if isinstance(cols, str):
        cols = [cols]

    if not isinstance(cols, list):
        raise TypeError('cols must be either a string or a list of strings')

    for col in cols:

        if not isinstance(col, str):
            raise TypeError('cols must be either a string or a list of strings')

        if col not in df.columns:
            raise KeyError(col + ' is not a column in the dataframe')


def not_value_in_series(series, values):
    """
    Raises ValueError if given values are not in a pandas.Series

    Args:
        series: pandas.Series
        values: list of values of the same data type as the series
    """
    if not isinstance(values, list):
        values = [values]

    for v in values:

        if v not in series.tolist():
            raise ValueError(f'the value {v} is not in the series {series.name}.')


def get_df_numeric_cols_list(df):
    """Return a list of the names of all numeric columns in the dataframe"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError('input df must be a pandas.DataFrame')

    allowed_types = [float, np.float64, int, np.int64]
    cols_list = []

    for col in df.columns:

        if df[col].dtype in allowed_types:
            cols_list = cols_list + [col]

    return cols_list


# def forest_plot(
#     df, dv=None, sort_values=True, standardize=False, figsize=(12, 9),
#     ylabel_size=12, xlabel_size=14, xaxis_title=''
# ):
#     """"""
#     df = df.copy()
#     fig, ax = plt.subplots(figsize=figsize)
#     ax.set_yticks(range(df.shape[0]));
# #     ax.set_yticklabels(df[dv], fontsize=ylabel_size);
#     ax.set_xlabel(xaxis_title, fontsize=xlabel_size)
#     ax.axvline(x=df.mean().mean(), color='k', linestyle='--');
#
#     df['row_mean'] = df.mean(axis=1)
#     df['row_std'] = df.std(axis=1)
#     df['row_sem'] = df['row_std'] / np.sqrt(df.shape[0])
#     df['ci_upper'] = df['row_mean'] + (1.96 * df['row_sem'])
#     df['ci_lower'] = df['row_mean'] - (1.96 * df['row_sem'])
#
#     if sort_values:
#         df.sort_values(by='row_mean', inplace=True)
#
#     df.reset_index(inplace=True, drop=True)
#
#     for row in range(df.shape[0]):
#         lower = df.loc[row, 'ci_lower']
#         upper = df.loc[row, 'ci_upper']
#         mean = df.loc[row, 'row_mean']
#         ax.plot([lower, upper], [row, row], color='k')
#         ax.plot([mean, mean], [row, row], marker='s', linestyle='', color='red')
#
#     ax.set_ylabel = df.index
#     ax.invert_yaxis()
#
#     return ax

