# data processing packages
import numpy as np
import pandas as pd

# stat packages
import scipy
from scipy import stats as st
from scipy.optimize import curve_fit
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.weightstats as smsw
from statsmodels.sandbox.stats import *
import statsmodels.stats.anova as anova #for ANOVA
from statsmodels.stats.multicomp import pairwise_tukeyhsd #for Tukey's multiple comparisons

# viz packages
import matplotlib as mpl
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
from matplotlib import colors
from matplotlib.lines import Line2D  
import seaborn as sns

# other utility packages
import json
import geopy
import ast
import datetime
import calendar
from dateutil.relativedelta import relativedelta

# import customized packages/functions
from .preprocessing_and_utilities import *


def generate_summary_report(data, groupby_cols, value_cols):

    # get the summary statistics
    summary_df = data.groupby(groupby_cols)[value_cols].median()
    summary_df.columns = ['median_visits_2019', 'median_visits_2020']
    summary_df = summary_df.dropna()
    summary_df = summary_df.astype(int)
    
    # get t test results for visits to individual parks in 2019 and 2020
    for index, row in summary_df.iterrows():
        data_sub = data.copy()
        
        for i, groupby_col in enumerate(groupby_cols):
            data_sub = data_sub[data_sub[groupby_col] == index[i]]

        test_result = scipy.stats.wilcoxon(data_sub['raw_visit_counts_2020'],
                                           data_sub['visits_base_adjtd'])

        
        summary_df.loc[index, 'n_sample'] = data_sub.shape[0]
        summary_df.loc[index, 'wilcoxon_t_stat'] = test_result[0]
        summary_df.loc[index, 'p_value'] = test_result[1]
        
    summary_df = summary_df.astype({'n_sample': 'int64', 'wilcoxon_t_stat': 'int64'})
    summary_df = summary_df.reset_index()
    summary_df = summary_df.sort_values(by=['borough', 'month', 'median_visits_2020'])


    return summary_df


def summary_by_topic(data, groupby_cols, value_cols, topic, test_method='wilcoxon', complete_report=False):
    
    # get the summary statistics
    summary_df = data.groupby(groupby_cols)[value_cols].sum()
    summary_df.columns = ['visits_2019', 'visits_2020']
    summary_df['visits_change_rate'] = (summary_df['visits_2020'] - summary_df['visits_2019']) / summary_df['visits_2019'] * 100
    summary_df['visits_change_rate'] = summary_df['visits_change_rate'].apply(lambda x: round(x, 1))
    
    # get t test results for visits to individual parks in 2019 and 2020
    for index, row in summary_df.iterrows():
        data_sub = data.copy()
        
        for i, groupby_col in enumerate(groupby_cols):
            data_sub = data_sub[data_sub[groupby_col] == index[i]]

        if test_method == 'paired_t':
            test_result = scipy.stats.ttest_rel(data_sub[value_cols[1]],
                                                data_sub[value_cols[0]])
        elif test_method == 'independent_t':
            test_result = scipy.stats.ttest_ind(data_sub[value_cols[1]],
                                                data_sub[value_cols[0]])

        elif test_method == 'wilcoxon':
            test_result = scipy.stats.wilcoxon(data_sub[value_cols[1]],
                                               data_sub[value_cols[0]])

        
        summary_df.loc[index, 'test_stat'] = test_result[0]
        summary_df.loc[index, 'p_value'] = test_result[1]


    if complete_report:
        
        summary_report = pd.pivot_table(data=summary_df.reset_index(), values=['visits_2019', 'visits_2020', 'visits_change_rate'], index=['month'], columns=[topic])
        summary_report.columns = summary_report.columns.swaplevel(0,1)
        summary_report.sort_index(axis=1, level=0, inplace=True)

        return summary_report

    return summary_df


def summary_by_topic_simple(data, topic_col, value_cols):

    # get the summary statistics
    data_sub = data[~data['month'].isin([1,2])]
    summary_df = data.groupby(topic_col)[value_cols].sum()
    summary_df.columns = ['visits_2019', 'visits_2020']
    summary_df['visits_change_rate'] = (summary_df['visits_2020'] - summary_df['visits_2019']) / summary_df['visits_2019'] * 100
    summary_df['visits_change_rate'] = summary_df['visits_change_rate'].apply(lambda x: round(x, 1))
    

    return summary_df


def plot_by_topic(datasets, labels, titles, topic_label, text_locs=[None,None,None,None], p_threshold=0.05):

    fig, axes = plt.subplots(figsize=(16,10), ncols=2, nrows=2, sharey=True)
    axes = axes.flatten()
    cmap = plt.cm.get_cmap('tab10', 10)

    text_loc_origs = []
    
    # construct a custom legend
    legend_heading = [Line2D([], [], linestyle='None', marker='None', label=topic_label)]
    legend_items = []

    for i, data in enumerate(datasets):

        # initialize a dataframe to record the original label locations
        text_loc_orig = pd.DataFrame(columns=['category', 'yloc_orig'])

        for j, category in enumerate(data.index.get_level_values(0).categories):
            
            # plot the visits_change_rate time series by topic
            data_sub = data.loc[category, ['visits_change_rate', 't_stat', 'p_value']].reset_index()
            axes[i].plot('month', 'visits_change_rate', '-o', ms=10, color=cmap.colors[j], data=data_sub, label=category + ' - ' + labels[i].loc[category, 'label'])
            
            # add each item to the custom legend (only add once)
            if i == 0:               
                legend_item = Line2D([0], [0], linestyle='None', marker='o', color=cmap.colors[j], ms=10, label=category)
                legend_items.append(legend_item)

            # ## vary marker symbol by paired t-test results
            # # insignificant
            # data_sub1 = data_sub[data_sub['p_value'] > p_threshold]
            # axes[i].scatter('month', 'visits_change_rate', marker='o', s=80, facecolors='none', edgecolors=cmap.colors[j], lw=2, data=data_sub1)

            # # significant
            # data_sub2 = data_sub[data_sub['p_value'] <= p_threshold]
            # axes[i].scatter('month', 'visits_change_rate', marker='o', s=80, color=cmap.colors[j], data=data_sub2)

            # # significant - negative
            # data_sub3 = data_sub[(data_sub['p_value'] <= p_threshold) & (data_sub['t_stat'] < 0)]
            # ax.scatter('month', 'visits_change_rate', marker=11, color=cmap.colors[i], data=data_sub3)

            # record the original label locations
            text_yloc_orig = data_sub[data_sub['month'] == 12]['visits_change_rate'].values[0]
            
            text_loc_orig.loc[j, 'category'] = category
            text_loc_orig.loc[j, 'yloc_orig'] = text_yloc_orig

            # plot the group labels using the manually adjusted label locations
            text = labels[i].loc[category, 'label']
            
            if text_locs[i] is not None:
                text_yloc = text_locs[i].loc[j, 'yloc_adjt']
            else:
                text_yloc = text_loc_orig.loc[j, 'yloc_orig']

            axes[i].text(12.7, text_yloc, s=text, fontsize=20, zorder=10, color=cmap.colors[j])


        # calculate the distance between the original label locations
        text_loc_orig = text_loc_orig.sort_values(by=['yloc_orig'], ascending=False)
        text_loc_orig['yloc_diff'] = text_loc_orig['yloc_orig'].diff()
        text_loc_origs.append(text_loc_orig)

        # mark March, when the pandemic started
        axes[i].axvline(3, ls='--', c='tab:red', alpha=0.5)

        # mark 0% visits change rate line
        axes[i].axhline(0, ls='--', c='grey', alpha=0.5)

        # set month names on the x-axis
        xticks = np.arange(1,13,1)
        xticklabels = [calendar.month_abbr[i] for i in xticks]

        # axes settings
        axes[i].set_title(titles[i], fontsize=20)
        axes[i].set_xlabel('Month', fontsize=18)
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels(xticklabels, fontsize=16)

        axes[0].set_ylabel('Percentage Change (%)', fontsize=18)
        axes[i].tick_params(axis='y', labelsize=16)
        axes[i].set_zorder(3)

    plt.tight_layout()

    # add the custom legend
    legend_elements = legend_heading + legend_items    # combine all legend elements
    plt.legend(handles=legend_elements, frameon=True, fancybox=True,
                fontsize=16, loc='upper left', bbox_to_anchor=(1.09, 2.2))


    return text_loc_origs


def generate_tukeyhsd_group_letters(res):

    # generate tukey_hsd group letters
    # this is a function to do Piepho method.  AN Alogrithm for a letter based representation of al-pairwise comparisons.  
    groups = res.groupsunique
    tot = len(groups)
    
    # make an empty dataframe that is a square matrix of size of the groups. #set first column to 1
    df_ltr = pd.DataFrame(np.nan, index=np.arange(tot), columns=np.arange(tot))
    df_ltr.iloc[:,0] = 1
    count = 0
    
    for i in np.arange(tot):   # loop through and make all pairwise comparisons. 
        for j in np.arange(i+1,tot):
            # print('i=',i,'j=',j,thsd.reject[count])
            if res.reject[count] == True:
                for cn in np.arange(tot):
                    if df_ltr.iloc[i,cn] == 1 and df_ltr.iloc[j,cn] == 1: # If the column contains both i and j shift and duplicat
                        
                        df_ltr = pd.concat([df_ltr.iloc[:,:cn+1], df_ltr.iloc[:,cn+1:].shift(axis=1)], axis=1)
                        df_ltr.iloc[:,cn+1] = df_ltr.iloc[:,cn]
                        df_ltr.iloc[i,cn] = 0
                        df_ltr.iloc[j,cn+1] = 0
                    
                    # check all columns for abosortpion.
                    for cleft in np.arange(len(df_ltr.columns)-1):
                        for cright in np.arange(cleft+1,len(df_ltr.columns)):
                            if (df_ltr.iloc[:,cleft].isna()).all() == False and (df_ltr.iloc[:,cright].isna()).all() == False: 
                                
                                if (df_ltr.iloc[:,cleft] >= df_ltr.iloc[:,cright]).all() == True:  
                                    
                                    df_ltr.iloc[:,cright] = 0
                                    df_ltr = pd.concat([df_ltr.iloc[:,:cright], df_ltr.iloc[:,cright:].shift(-1,axis=1)],axis=1)
                                
                                if (df_ltr.iloc[:,cleft] <= df_ltr.iloc[:,cright]).all() == True:
                                    
                                    df_ltr.iloc[:,cleft] = 0
                                    df_ltr = pd.concat([df_ltr.iloc[:,:cleft], df_ltr.iloc[:,cleft:].shift(-1,axis=1)],axis=1)

            count+=1

    # sort values to make the first column to become A        
    df_ltr = df_ltr.sort_values(by=list(df_ltr.columns), axis=1, ascending=False)

    # assign letters to each column
    for cn in np.arange(len(df_ltr.columns)):
        
        df_ltr.iloc[:,cn] = df_ltr.iloc[:,cn].replace(1,chr(97+cn)) 
        df_ltr.iloc[:,cn] = df_ltr.iloc[:,cn].replace(0,'')
        df_ltr.iloc[:,cn] = df_ltr.iloc[:,cn].replace(np.nan,'') 

    # put all the letters into one string
    df_ltr = df_ltr.astype(str)
    group_letters = df_ltr.sum(axis=1).to_list()
    tukeyhsd_labels = pd.DataFrame(data=group_letters, index=groups, columns=['label'])
    

    return tukeyhsd_labels



def tukeyhsd_test(data, value_col, topic, subtopic=None, filter_month=False):

    summary_sub = data.reset_index()
    if filter_month == True:
        summary_sub = summary_sub[~summary_sub['month'].isin([1,2])]    # drop the first two months when the pandemic didn't start

    if subtopic is not None:
        
        # initialize dictionaries to save results for each subtopic category
        res_dfs = {}
        group_letters = {}    
        
        for category in summary_sub[subtopic].unique():
            summary_sub1 = summary_sub[summary_sub[subtopic] == category]

            res = pairwise_tukeyhsd(summary_sub1[value_col], summary_sub1[topic])
            res_df = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
            res_dfs[category] = res_df

            # generate tukey_hsd group letters
            group_letters[category] = generate_tukeyhsd_group_letters(res)

        return res_dfs, group_letters

    else:

        res = pairwise_tukeyhsd(summary_sub[value_col], summary_sub[topic])
        res_df = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])

        # generate tukey_hsd group letters
        group_letters = generate_tukeyhsd_group_letters(res)
    
        return res_df, group_letters
    


def label_income_group(data, income_data, income_theme, n_group, origin_dest=''):
        
    bins = [income_data[income_theme].quantile(i/n_group) for i in range(n_group+1)]
    
    if n_group == 3:
        data['income_group'] = pd.cut(data[income_theme + origin_dest],
                                      bins=bins, labels=['lower', 'middle', 'upper'])        
    elif n_group == 4:  
        data['income_group'] = pd.cut(data[income_theme + origin_dest],
                                      bins=bins, labels=['lower', 'lower middle', 'upper middle', 'upper'])   
    elif n_group == 5: 
        data['income_group'] = pd.cut(data[income_theme + origin_dest],
                                      bins=bins, labels=['lower', 'lower middle', 'middle', 'upper middle', 'upper'])
        
    return data


def plot_by_ses_and_topic(summary_data, topic_categories, cat_col, value_col, ylabel, tukeyhsd_labels, title=None):
    
    # initialize the subplots by the number of topic categories
    num_of_topic_categories = len(topic_categories)
    if num_of_topic_categories == 8:
        fig, axes = plt.subplots(figsize=(16,8), nrows=2, ncols=4)
    elif num_of_topic_categories == 9:
        fig, axes = plt.subplots(figsize=(13,13), nrows=3, ncols=3)
    
    axes = axes.flatten()

    data = summary_data.reset_index()
    xticks = np.arange(2,14,2)
    xticklabels = [calendar.month_abbr[i] for i in xticks]


    for i, park_type in enumerate(topic_categories):

        for j, cat in enumerate(data[cat_col].unique()):
            data_sub = data[(data['park_type']==park_type) & (data[cat_col]==cat)]
            label = tukeyhsd_labels[park_type].loc[cat,'label'] + ' - ' + str(cat)
            axes[i].plot('month', value_col, 'o-', label=label, data=data_sub)

        # mark March, when the pandemic started
        axes[i].axvline(2.9, ls='--', c='tab:red', alpha=0.5)

        # mark 0% visits change rate line
        axes[i].axhline(0, ls='--', c='grey', alpha=0.5)
        
        # axes settings
        axes[i].set_title(park_type, fontsize=14)
        axes[i].set_xlabel('Month', fontsize=13)
        axes[i].set_xticks(xticks)
        axes[i].set_xticklabels(xticklabels, fontsize=13)
        axes[i].set_ylabel(ylabel, fontsize=13)
        axes[i].tick_params(axis='y', labelsize=13)
        axes[i].legend(fontsize=14, labelcolor='linecolor')

        

    # set the y-axis of all subplots to have the same range
    y_max = data[value_col].to_numpy().max()
    y_min = data[value_col].to_numpy().min()
    y_range = y_max - y_min
    
    if title is not None:
        plt.suptitle(title, fontsize=15)
    plt.setp(axes, ylim = (y_min - y_range*0.1, y_max + y_range*0.1))
    plt.tight_layout();



### generate summary tables
def summarize_travel_distance(data, topic_col, value_col, by_income_group=False):

    # subselect data
    if not by_income_group:
        data_sub = data[[topic_col, value_col, 'month', 'year']]
    else:
        data_sub = data[[topic_col, value_col, 'income_group', 'month', 'year']]

    data_sub = data_sub.dropna()
    data_sub = data_sub[~data_sub['month'].isin([1,2])]    # drop the first two months when the pandemic didn't start
    data_before_all = data_sub[data_sub['year'] == 2019]
    data_during_all = data_sub[data_sub['year'] == 2020]

    # initialize a list to store summary_dfs
    summary_dfs = []

    # add an Overall category to the topic list
    topic_cat = list(data_sub[topic_col].unique())
    topic_cat.append('Overall')


    def calculate_summary_stats(data_before, data_during, topic_col, value_col, topic_cat, income_group):

        # initialize the summary df
        summary_df = pd.DataFrame()

        # calculate the mean
        summary_df['before'] = data_before.groupby(topic_col)[value_col].mean()
        summary_df['during'] = data_during.groupby(topic_col)[value_col].mean()
        
        summary_df.loc['Overall','before'] = data_before[value_col].mean()
        summary_df.loc['Overall','during'] = data_during[value_col].mean()
        
        summary_df['mean_diff'] = summary_df['during'] - summary_df['before']

        for i, cat in enumerate(topic_cat):
            
            if cat != 'Overall':

                d1 = smsw.DescrStatsW(data_during[data_during[topic_col] == cat][value_col].dropna().to_numpy())
                d2 = smsw.DescrStatsW(data_before[data_before[topic_col] == cat][value_col].dropna().to_numpy())
            
            else:

                d1 = smsw.DescrStatsW(data_during[value_col].dropna().to_numpy())
                d2 = smsw.DescrStatsW(data_before[value_col].dropna().to_numpy())

            # calculate the confidence interval (lower boundary and upper boundary)
            lb, ub = smsw.CompareMeans(d1,d2).tconfint_diff(usevar='unequal')

            summary_df.loc[cat,'diff_lb'] = lb
            summary_df.loc[cat,'diff_ub'] = ub

        # calculate the percentage changes
        summary_df['mean_diff_pct'] = summary_df['mean_diff'] / summary_df['before'] * 100
        summary_df['diff_lb_pct'] = summary_df['diff_lb'] / summary_df['before'] * 100
        summary_df['diff_ub_pct'] = summary_df['diff_ub'] / summary_df['before'] * 100
        
        # label the income group(s)
        summary_df['income_group'] = income_group
        summary_df = summary_df.reset_index()

        return summary_df


    def label_groups(summary_df, data_before, data_during, topic_col, value_col, by_income_group):

        # compare and generate group labels by topic_col
        if not by_income_group:

            data_before_tukeyhsd = data_before.groupby([topic_col,'month'])[value_col].mean()
            res_dfs_b, group_letters_b = tukeyhsd_test(data_before_tukeyhsd, value_col, topic_col)
            summary_df['groups_before'] = group_letters_b

            data_during_tukeyhsd = data_during.groupby([topic_col,'month'])[value_col].mean()
            res_dfs_d, group_letters_d = tukeyhsd_test(data_during_tukeyhsd, value_col, topic_col)
            summary_df['groups_during'] = group_letters_d

        # compare and generate group labels by income_group
        else:
            
            summary_df = summary_df.set_index([topic_col,'income_group'])

            # initialize empty columns to store group letters
            summary_df['groups_before'] = ""
            summary_df['groups_during'] = ""

            for cat in topic_cat:
                
                if cat != 'Overall':

                    data_before_tukeyhsd = data_before[data_before[topic_col] == cat]
                    res_dfs_b, group_letters_b = tukeyhsd_test(data_before_tukeyhsd, value_col, 'income_group')
                    
                    data_during_tukeyhsd = data_during[data_during[topic_col] == cat]
                    res_dfs_b, group_letters_d = tukeyhsd_test(data_during_tukeyhsd, value_col, 'income_group')
                    
                else:
                    
                    res_dfs_b, group_letters_b = tukeyhsd_test(data_before, value_col, 'income_group')
                    res_dfs_b, group_letters_d = tukeyhsd_test(data_during, value_col, 'income_group')

                for income_group in ['lower','middle','upper']:
                    
                    summary_df.loc[(cat, income_group), 'groups_before'] = group_letters_b.loc[income_group, 'label']
                    summary_df.loc[(cat, income_group), 'groups_during'] = group_letters_d.loc[income_group, 'label']
        

        summary_df = summary_df.reset_index()

        return summary_df
    

    # summarize by 
    if not by_income_group:
        
        summary_df_all = calculate_summary_stats(data_before_all, data_during_all, topic_col, value_col, topic_cat, "All")
        summary_df_all = label_groups(summary_df_all, data_before_all, data_during_all, topic_col, value_col, by_income_group)

        summary_df = summary_df_all.copy()

    # summarize by income group
    else:

        for income_group in ['lower','middle','upper']:
            
            data_before_ig = data_before_all[data_before_all['income_group'] == income_group]
            data_during_ig = data_during_all[data_during_all['income_group'] == income_group]
            
            summary_df_ig = calculate_summary_stats(data_before_ig, data_during_ig, topic_col, value_col, topic_cat, income_group)
            
            summary_dfs.append(summary_df_ig)
    
        summary_df = pd.concat(summary_dfs, ignore_index=True)
        summary_df = label_groups(summary_df, data_before_all, data_during_all, topic_col, value_col, by_income_group)


    return summary_df



### generate summary graphs
def travel_distance_summary_graph_overall(data_summary, topic_col, width, title, xlabel, xlim):

    fig, axes = plt.subplots(figsize=(10,4), ncols=2, sharey=True)

    data_summary_melt = data_summary.melt(id_vars=[topic_col], value_vars=['before', 'during'], var_name='year', value_name='mean_dist')
    data_summary_melt['year'] = data_summary_melt['year'].replace({'before': 2019, 'during':2020})

    # barplot for mean travel distance in 2019 and 2020
    sns.barplot(data=data_summary_melt, y=topic_col, x="mean_dist", width=width, hue='year', ax=axes[0])
    
    axes[0].set_title('(a) ' + title)
    axes[0].set_xlabel(xlabel)
    axes[0].set_xlim(xlim)
    axes[0].set_ylabel('Park Type')

    # barplot for percentage change of mean travel distance
    axes[1] = sns.barplot(data=data_summary, y=topic_col, x="mean_diff_pct", color='tab:orange', width=width)

     # add errorbars from the pre-calculated 95% CIs
    yticks = axes[1].get_yticks()    # the y center locations
    axes[1].errorbar(x=data_summary["mean_diff_pct"], y=yticks, xerr=data_summary["mean_diff_pct_cim"], ls='', lw=1, color='black')

    axes[1].set_title('(b) Percentage Change of ' + title)
    axes[1].set_xlabel('Percentage Change (%)')
    axes[1].set_ylabel('')

    plt.tight_layout();


def travel_distance_summary_graph(data_summary, topic_col, hue_col, width):

    # subselect data
    # data_sub = data[~data['month'].isin([1,2])]    # drop the first two months when the pandemic didn't start
    # data_2019 = data_sub[data_sub['year'] == 2019]
    # data_2020 = data_sub[data_sub['year'] == 2020]

    ## barplots for mean travel distance in 2019 and 2020
    fig, axes = plt.subplots(figsize=(12,4), ncols=3, sharey=True)

    # sns.boxplot(data=data_2019, y=topic_col, x=value_col, hue=hue_col, showfliers=False,
    #             linewidth=1, notch=True, showcaps=False, ax=axes[0])
    # sns.boxplot(data=data_2020, y=topic_col, x=value_col, hue=hue_col, showfliers=False,
    #             linewidth=1, notch=True, showcaps=False, ax=axes[1])

    income_groups = ['lower','middle','upper']
    
    sns.barplot(data=data_summary, y=topic_col, x="before", width=width, hue=hue_col, ax=axes[0])
    sns.barplot(data=data_summary, y=topic_col, x="during", width=width, hue=hue_col, ax=axes[1])

    years = [2019, 2020]
    label_cols = ['groups_before', 'groups_during']
    for i, year in enumerate(years):

        axes[i].set_title('(' + chr(97+i) + ') ' + 'Mean Travel Distance in ' + str(year))
        axes[i].set_xlabel('Travel Distance (km)')
        axes[i].set_xlim(0,11)

        for j, container in enumerate(axes[i].containers):
            
            # set the bar label
            labels = data_summary[data_summary['income_group'] == income_groups[j]][label_cols[i]]
            axes[i].bar_label(container, labels=labels, padding=3)

    ## barplot for percentage change of mean travel distance
    axes[2] = sns.barplot(data=data_summary, y=topic_col, x="mean_diff_pct", width=width, hue=hue_col)

    # add errorbars from the pre-calculated 95% CIs
    yticks = axes[2].get_yticks()    # the y center locations
    yoffsets = [-width/3, 0, width/3]

    for i, income_group in enumerate(income_groups):
        
        data_summary_sub = data_summary[data_summary['income_group'] == income_group]
        ylocs = yticks + yoffsets[i]
        axes[2].errorbar(x=data_summary_sub["mean_diff_pct"], y=ylocs, xerr=data_summary_sub["mean_diff_pct_cim"], ls='', lw=1, color='black')

    axes[2].set_title('(c) Percentage Change of Mean Travel Distance')
    axes[2].set_xlabel('Percentage Change (%)')

    axes[0].set_ylabel("Park Type")
    axes[0].legend('', frameon=False)
    axes[1].set_ylabel("")
    axes[2].set_ylabel("")
    axes[2].legend('', frameon=False)



    plt.tight_layout();

