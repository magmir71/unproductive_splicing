from __future__ import (absolute_import, division,
                        print_function, unicode_literals)

import psycopg2
import pandas.io.sql as sqlio
# Макарыч
conn = psycopg2.connect(database='gtex_ipsa', user='postgres', host='localhost', port='5439')

import warnings
warnings.simplefilter('ignore')
import pandas as pd
import numpy as np
from numpy import *
import os
import json
import time
import scipy.stats as stats
import statsmodels.stats as smstats
import statsmodels.api as sm
from multiprocessing import Process, Manager, Pool
import multiprocessing
from functools import partial
from collections import Counter
import seaborn as sns; sns.set()
import matplotlib
matplotlib.style.use('seaborn')
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.rcParams['backend'] = "Qt5Agg"
import matplotlib.ticker as ticker
from matplotlib.ticker import FuncFormatter
from IPython.display import display, Image

from statsmodels.stats.multitest import multipletests

def get_bins(a,b,step=2):
    t = []
    LQ=a['psi'].max()*100
    UQ=b['psi'].min()*100
    p=-1
    area = 'LQ'
    t.append(p/100)
    while p<=100:
        if p>=LQ and area=='LQ':
            area = 'MID'
            if abs(LQ-p)>=p*0.2:
                t.pop()
                t.append(LQ/100+0.0001)
                p=LQ+step
            else:
                t.append(LQ/100+0.0001)
                p=p+step                
        elif p>=UQ and area=='MID':
            area = 'UQ'
            if abs(UQ-p)<=p*0.2:
                t.pop()
                t.append(UQ/100-0.0001)
                p=UQ+step
            else:
                t.append(UQ/100-0.0001)
                p=p+step   
        else:
            t.append(p/100)
            p=p+step
    return t

def get_pvalue_star(pval,thr=0.05):
    if thr==0.05:
        if pval<0.001:
            return '***'
        elif pval<0.01:
            return '**'
        elif pval<0.05:
            return '*'
        else:
            return 'NS'
    elif thr==0.1:
        if pval<0.001:
            return '***'
        elif pval<0.01:
            return '**'
        elif pval<0.1:
            return '*'
        else:
            return 'NS'
        
def get_isoform_number(x):
    if 'NMD' in x['isoform']:
        return 1
    else:
        return 2

def get_psi_data_gtex(target,NMD_cols,can_cols):
    number_of_gtex_samples = 8551
    
    a = []
    input_dfs = []
    junctions_to_load = []
    sites_to_load = []
    genes_to_load = []
    
    row = target.iloc[0]
    
    gene_name = row['Target']
    AS_event_position = row['AS event position']

    input_df = pd.DataFrame([list(row[NMD_cols+can_cols]),
                          list(NMD_cols+can_cols)]).transpose()
    input_df.columns=['long_id','isoform']
    input_df['number'] = input_df.apply(lambda x:get_isoform_number(x),1)
    input_df['isoform'] = input_df['isoform']+'_'+input_df['number'].astype('str')
    input_df = input_df.loc[input_df['long_id']!='']
    input_df['gene_name'] = gene_name
    input_df['AS_event_position'] = AS_event_position

    input_dfs.append(input_df)

    junctions_to_load = junctions_to_load+list(input_df.loc[input_df['isoform'].str.contains('junction')]['long_id'])
    sites_to_load = sites_to_load+list(input_df.loc[input_df['isoform'].str.contains('site')]['long_id'])
    genes_to_load = genes_to_load+[gene_name]

    junctions_to_load = list(pd.Series(junctions_to_load).unique())
    sites_to_load = list(pd.Series(sites_to_load).unique())
    genes_to_load = list(pd.Series(genes_to_load).unique())

    input_dfs = pd.concat(input_dfs)
    input_dfs = input_dfs.reset_index(drop=True)

    input_dfs_j = input_dfs.loc[~input_dfs['isoform'].str.contains('site')]
    input_dfs_s = input_dfs.loc[input_dfs['isoform'].str.contains('site')]    

    sql = """SELECT junction_long_id, junction_counts FROM all_junction_counts_A04 
            INNER JOIN gtex_junctions_A06 USING (id)
        WHERE junction_long_id IN ("""+"'"+"','".join(junctions_to_load)+"'"+""")
        """
    dat_j = pd.read_sql_query(sql, conn)
    dat_j = pd.merge(input_dfs_j,dat_j.rename(columns={'junction_long_id':'long_id','junction_counts':'counts'}),how='left',on='long_id')
    dat_j['counts'] = '['+(dat_j['counts'].str.replace('\t',', '))+']'

    sql = """SELECT site_long_id, site_counts FROM all_sites_counts_A04 
                INNER JOIN gtex_sites_a06 ON id = site_id
            WHERE site_long_id IN ("""+"'"+"','".join(sites_to_load)+"'"+""")
            """
    dat_s = pd.read_sql_query(sql, conn)
    dat_s = pd.merge(input_dfs_s,dat_s.rename(columns={'site_long_id':'long_id','site_counts':'counts'}),how='left',on='long_id')

    dat = pd.concat([dat_j,dat_s])
    dat = dat.sort_values(['gene_name','AS_event_position','number','isoform']).reset_index(drop=True)

    sql = """SELECT gene_name, gene_counts FROM gene_counts_hg19 
            WHERE gene_name IN ("""+"'"+"','".join(genes_to_load)+"'"+""")
            """
    dat_g = pd.read_sql_query(sql, conn)
    expr = dat_g['gene_counts'].astype('str').str[1:-1].str.split(', ',expand=True).transpose()
    expr.columns = list(dat_g['gene_name'])
    expr = expr.astype('int')
    expr['sample_id'] = expr.index+1
    del dat_g

    Exclude_tissues = ['Cells - Transformed fibroblasts','Cells - EBV-transformed lymphocytes']

    sample_metadata = pd.read_csv('/home/magmir/TASS/GTEX/sample_metadata.tsv',delimiter="\t",
                                   index_col=None,header=0)
    sample_metadata = sample_metadata.loc[(sample_metadata['sf']>0)&(
        sample_metadata['sf_global']>0)&(
        ~sample_metadata['smtsd'].isin(Exclude_tissues))]

    gr = dat[['gene_name','AS_event_position']].drop_duplicates()
    i=0
    for index, row in gr.iterrows():
        if row['gene_name'] not in expr.columns:
            print('no expression for: '+row['gene_name']+' '+row['AS_event_position'])
            i=i+1
            continue
        temp = dat.loc[(dat['gene_name']==row['gene_name'])&(dat['AS_event_position']==row['AS_event_position'])]
        if len(temp)==0:
            print('no data for: '+row['gene_name']+' '+row['AS_event_position'])
            i=i+1
            continue
        temp['counts'] = temp['counts'].fillna('['+(', '.join(['0']*number_of_gtex_samples))+']')
        data = temp['counts'].astype('str').str[1:-1].str.split(', ',expand=True).transpose()
        data.columns = list(temp['isoform'])
        data['sample_id'] = data.index+1
        data = pd.merge(data,sample_metadata[['sample_id','smtsd','sf','sf_global']],
                          how='inner',on='sample_id')
        data[list(temp['isoform'])] = data[list(temp['isoform'])].fillna(0).astype('int')
        data['denom'] = data[list(temp.loc[temp['number']==1]['isoform'])].mean(1)
        for oiso in list(temp.loc[temp['number']!=1]['number'].unique()):
            data['denom']=data['denom']+data[list(temp.loc[temp['number']==oiso]['isoform'])].mean(1)
        data['psi'] = (data[list(temp.loc[temp['number']==1]['isoform'])].mean(1))/(data['denom'])
        data['t']=1
        df_gr = data.groupby('smtsd').agg({'t':sum}).reset_index()
        data = data.loc[(~data['psi'].isna())&(data['denom']>15)]
        # порог на знаменатель - 15
        # если внутри ткани оказалось удалено более 80% образцов, то надо убрать такую ткань полностью
        df_gr_1 = data.groupby('smtsd').agg({'t':sum}).reset_index().rename(columns={'t':'t1'})
        df_gr_1 = pd.merge(df_gr_1,df_gr,how='left',on='smtsd')
        df_gr_1['perc_deleted'] = ((df_gr_1['t']-df_gr_1['t1'])/df_gr_1['t']*100)
        df_gr_1['remove'] = ((df_gr_1['perc_deleted']>80)).astype('int')
        data = pd.merge(data,df_gr_1[['smtsd','remove']],how='left',on='smtsd')
        data['remove'] = data['remove'].fillna(0).astype('int')
        data = data.loc[data['remove']==0]
        data = data.reset_index(drop=True)
        data.drop(['t','remove'],1,inplace=True)

        # разделяем по квантилям psi
        thrL = data['psi'].quantile(0.25)
        thrR = data['psi'].quantile(0.75)
        psi_median = data['psi'].median()
        data['cluster'] = ((data['psi']<=thrL).astype('int')*(-1)+(data['psi']>=thrR).astype('int')).astype('str').str.replace('-1','LQ').replace('1','UQ').replace('0','MED')    

        # добавляем экспрессию гена
        data = pd.merge(data,expr[['sample_id',row['gene_name']]],how='inner',on=['sample_id'])
        data[row['gene_name']] = np.log2((1/data['sf_global'])*(data[row['gene_name']]+8))
        data[row['gene_name']] = data[row['gene_name']]-data[row['gene_name']].median()
        data['global_expr'] = data[row['gene_name']]

        data['local_expr'] = np.log2((1/data['sf'])*(data['denom']))
        data['local_expr'] = data['local_expr']-data['local_expr'].median()
        
    return data
    
def get_GTEX_summary_pooled(targets,NMD_cols,can_cols,return_aggregated_data = False,Exclude_tissues = ['Cells - Transformed fibroblasts','Cells - EBV-transformed lymphocytes','Testis']):
    number_of_gtex_samples = 8551
    
    a = []
    input_dfs = []
    junctions_to_load = []
    sites_to_load = []
    genes_to_load = []
    try:
        i=0
        for index, row in targets.iterrows():
            gene_name = row['Target']
                
            AS_event_position = row['AS event position']

            input_df = pd.DataFrame([list(row[NMD_cols+can_cols]),
                                  list(NMD_cols+can_cols)]).transpose()
            input_df.columns=['long_id','isoform']
            input_df['number'] = input_df.apply(lambda x:get_isoform_number(x),1)
            input_df['isoform'] = input_df['isoform']+'_'+input_df['number'].astype('str')
            input_df = input_df.loc[input_df['long_id']!='']
            input_df['gene_name'] = gene_name
            input_df['AS_event_position'] = AS_event_position

            input_dfs.append(input_df)

            junctions_to_load = junctions_to_load+list(input_df.loc[input_df['isoform'].str.contains('junction')]['long_id'])
            sites_to_load = sites_to_load+list(input_df.loc[input_df['isoform'].str.contains('site')]['long_id'])
            
            if gene_name=='CCN1':
                genes_to_load = genes_to_load+['CYR61']
            else:
                genes_to_load = genes_to_load+[gene_name]
            i=i+1

        junctions_to_load = list(pd.Series(junctions_to_load).unique())
        sites_to_load = list(pd.Series(sites_to_load).unique())
        genes_to_load = list(pd.Series(genes_to_load).unique())

        input_dfs = pd.concat(input_dfs)
        input_dfs = input_dfs.reset_index(drop=True)

        input_dfs_j = input_dfs.loc[~input_dfs['isoform'].str.contains('site')]
        input_dfs_s = input_dfs.loc[input_dfs['isoform'].str.contains('site')]    

        sql = """SELECT junction_long_id, junction_counts FROM all_junction_counts_A04 
                INNER JOIN gtex_junctions_A06 USING (id)
            WHERE junction_long_id IN ("""+"'"+"','".join(junctions_to_load)+"'"+""")
            """
        dat_j = pd.read_sql_query(sql, conn)
        dat_j = pd.merge(input_dfs_j,dat_j.rename(columns={'junction_long_id':'long_id','junction_counts':'counts'}),how='left',on='long_id')
        dat_j['counts'] = '['+(dat_j['counts'].str.replace('\t',', '))+']'

        sql = """SELECT site_long_id, site_counts FROM all_sites_counts_A04 
                    INNER JOIN gtex_sites_a06 ON id = site_id
                WHERE site_long_id IN ("""+"'"+"','".join(sites_to_load)+"'"+""")
                """
        dat_s = pd.read_sql_query(sql, conn)
        dat_s = pd.merge(input_dfs_s,dat_s.rename(columns={'site_long_id':'long_id','site_counts':'counts'}),how='left',on='long_id')

        dat = pd.concat([dat_j,dat_s])
        dat = dat.sort_values(['gene_name','AS_event_position','number','isoform']).reset_index(drop=True)

        sql = """SELECT gene_name, gene_counts FROM gene_counts_hg19 
                WHERE gene_name IN ("""+"'"+"','".join(genes_to_load)+"'"+""")
                """
        dat_g = pd.read_sql_query(sql, conn)
        expr = dat_g['gene_counts'].astype('str').str[1:-1].str.split(', ',expand=True).transpose()
        cols = []
        
        for gene_name in list(dat_g['gene_name']):
            if gene_name=='CYR61':
                cols.append('CCN1')
            else:
                cols.append(gene_name)
        expr.columns = cols  
        
        for gene_name in list(targets['Target'].unique()):
            if gene_name not in list(expr.columns):
                print('no expression for '+gene_name)
        
        expr = expr.astype('int')
        expr['sample_id'] = expr.index+1
        del dat_g

        sample_metadata = pd.read_csv('/home/magmir/TASS/GTEX/sample_metadata.tsv',delimiter="\t",
                                       index_col=None,header=0)
        sample_metadata = sample_metadata.loc[(sample_metadata['sf']>0)&(
            sample_metadata['sf_global']>0)&(
            ~sample_metadata['smtsd'].isin(Exclude_tissues))]

        summary = []
        if return_aggregated_data:
            aggregated_data = []
        gr = dat[['gene_name','AS_event_position']].drop_duplicates()
        i=0
        for index, row in gr.iterrows():
            if row['gene_name'] not in expr.columns:
                i=i+1
                continue
            temp = dat.loc[(dat['gene_name']==row['gene_name'])&(dat['AS_event_position']==row['AS_event_position'])]
            if len(temp)==0:
                print('no psi data for: '+row['gene_name']+' '+row['AS_event_position'])
                i=i+1
                continue
            temp['counts'] = temp['counts'].fillna('['+(', '.join(['0']*number_of_gtex_samples))+']')
            data = temp['counts'].astype('str').str[1:-1].str.split(', ',expand=True).transpose()
            data.columns = list(temp['isoform'])
            data['sample_id'] = data.index+1
            data = pd.merge(data,sample_metadata[['sample_id','smtsd','sf','sf_global']],
                              how='inner',on='sample_id')
            data[list(temp['isoform'])] = data[list(temp['isoform'])].fillna(0).astype('int')
            data['denom'] = data[list(temp.loc[temp['number']==1]['isoform'])].mean(1)
            for oiso in list(temp.loc[temp['number']!=1]['number'].unique()):
                data['denom']=data['denom']+data[list(temp.loc[temp['number']==oiso]['isoform'])].mean(1)
            data['psi'] = (data[list(temp.loc[temp['number']==1]['isoform'])].mean(1))/(data['denom'])
            data['t']=1
            df_gr = data.groupby('smtsd').agg({'t':sum}).reset_index()
            data = data.loc[(~data['psi'].isna())&(data['denom']>15)]
            # порог на знаменатель - 15
            # если внутри ткани оказалось удалено более 80% образцов, то надо убрать такую ткань полностью
            df_gr_1 = data.groupby('smtsd').agg({'t':sum}).reset_index().rename(columns={'t':'t1'})
            df_gr_1 = pd.merge(df_gr_1,df_gr,how='left',on='smtsd')
            df_gr_1['perc_deleted'] = ((df_gr_1['t']-df_gr_1['t1'])/df_gr_1['t']*100)
            df_gr_1['remove'] = ((df_gr_1['perc_deleted']>80)).astype('int')
            data = pd.merge(data,df_gr_1[['smtsd','remove']],how='left',on='smtsd')
            data['remove'] = data['remove'].fillna(0).astype('int')
            data = data.loc[data['remove']==0]
            data = data.reset_index(drop=True)
            data.drop(['t','remove'],1,inplace=True)

            # разделяем по квантилям psi
            thrL = data['psi'].quantile(0.25)
            thrR = data['psi'].quantile(0.75)
            psi_median = data['psi'].median()
            data['cluster'] = ((data['psi']<=thrL).astype('int')*(-1)+(data['psi']>=thrR).astype('int')).astype('str').str.replace('-1','LQ').replace('1','UQ').replace('0','MED')    

            # добавляем экспрессию гена
            data = pd.merge(data,expr[['sample_id',row['gene_name']]],how='inner',on=['sample_id'])
            data[row['gene_name']] = np.log2((1/data['sf_global'])*(data[row['gene_name']]+8))
            data[row['gene_name']] = data[row['gene_name']]-data[row['gene_name']].median()
            data['global_expr'] = data[row['gene_name']]

            data['local_expr'] = np.log2((1/data['sf'])*(data['denom']))
            data['local_expr'] = data['local_expr']-data['local_expr'].median()
            
            if return_aggregated_data:
                data['gene_name'] = row['gene_name']
                data['AS_event_position'] = row['AS_event_position']
                aggregated_data.append(data[['gene_name','AS_event_position','sample_id','smtsd','psi','cluster','global_expr','local_expr']])
            
            # сравниваем квантили
            a = data.loc[data['cluster']=='LQ'][['global_expr','local_expr','psi']]
            b = data.loc[data['cluster']=='UQ'][['global_expr','local_expr','psi']]

            if len(a)>0 and len(b)>0:
                # в этом месте z-score считается неправильно! stats.mannwhitneyu[0] сообщает U1, ассоциированный с первой выборкой (a['global_expr'])  
                stat = stats.mannwhitneyu(a['global_expr'],b['global_expr'],alternative='two-sided')[0]
                zscore_global = (stat-len(a)*len(b)/2)/(len(a)*len(b)*(len(a)+len(b)+1)/12)**0.5
                stat = stats.mannwhitneyu(a['local_expr'],b['local_expr'],alternative='two-sided')[0]
                zscore_local = (stat-len(a)*len(b)/2)/(len(a)*len(b)*(len(a)+len(b)+1)/12)**0.5

                log2FC_global = (b['global_expr'].median()-a['global_expr'].median())
                log2FC_local = (b['local_expr'].median()-a['local_expr'].median())
                psi_b = b['psi'].median()
                psi_a = a['psi'].median()
                summary.append([row['gene_name'],row['AS_event_position'],psi_a,psi_b,log2FC_global,log2FC_local,zscore_global,zscore_local])
                if i%20==0:
                    print('done: '+str(i)+' out of '+str(len(gr)))
                i=i+1
            else:
                print('LQ or UQ are zero length: '+row['gene_name']+' '+row['AS_event_position'])
                i=i+1
                continue
        if return_aggregated_data:
            return summary,aggregated_data
        else:
            return summary
    except Exception:
        if row is not None:
            print('error in '+row['gene_name']+' '+row['AS_event_position'])
    
# получаем распределение psi в тканях GTEX
def get_psi_in_gtex(input_df,Add_Exclude_tissues=[]):
    sample_metadata = pd.read_csv('/home/magmir/TASS/GTEX/sample_metadata.tsv',delimiter="\t",
                                   index_col=None,header=0)
    sample_metadata = sample_metadata.loc[(sample_metadata['sf']>0)&(sample_metadata['sf_global']>0)]
    # если среди изоформ есть site, значит нужно выделить отдельно информацию по intron retention
    if len(input_df.loc[input_df['isoform'].str.contains('site')])>0:
        input_df_j = input_df.loc[~input_df['isoform'].str.contains('site')]
        input_df_s = input_df.loc[input_df['isoform'].str.contains('site')]
        
        sql = """SELECT * FROM all_sites_counts_A04 
                    INNER JOIN gtex_sites_a06 USING (id)
                WHERE site_long_id IN ("""+"'"+"','".join(input_df_s['long_id'])+"'"+""")
                """
        dat = pd.read_sql_query(sql, conn)
        dat = pd.merge(input_df_s,dat.rename(columns={'site_long_id':'long_id'}),how='left',on='long_id')

        site_count_data = dat['site_counts'].str.split('\t',expand=True).transpose()
        site_count_data.columns = list(dat['isoform'])
        site_count_data['sample_id'] = site_count_data.index+1    
    else:
        input_df_j = input_df.copy()
    sql = """SELECT * FROM all_junction_counts_A04 
                INNER JOIN gtex_junctions_A06 USING (id)
            WHERE junction_long_id IN ("""+"'"+"','".join(input_df_j['long_id'])+"'"+""")
            """
    dat = pd.read_sql_query(sql, conn)
    dat = pd.merge(input_df_j,dat.rename(columns={'junction_long_id':'long_id'}),how='left',on='long_id')
    
    junction_count_data = dat['junction_counts'].str.split('\t',expand=True).transpose()
    junction_count_data.columns = list(dat['isoform'])
    junction_count_data['sample_id'] = junction_count_data.index+1
    
    junction_count_data = pd.merge(junction_count_data,
                      sample_metadata[['sample_id','smtsd','sf','sf_global']],
                      how='inner',on='sample_id')
    
    Exclude_tissues = ['Cells - Transformed fibroblasts','Cells - EBV-transformed lymphocytes']+Add_Exclude_tissues
    junction_count_data = junction_count_data.loc[~junction_count_data['smtsd'].isin(Exclude_tissues)]
    # если был создан site_count_data, то присоединяем его
    if len(input_df.loc[input_df['isoform'].str.contains('site')])>0:
        junction_count_data = pd.merge(junction_count_data,site_count_data,how='left',on='sample_id')
    # считаем psi в образцах и выкидываем случаи, где psi не определено (деление на 0)
    # ожидаем, что в названиях изоформ в конце указано _1, _2, _3 ..., psi считаем как _1/(_1+_2+_3)
    isoforms = pd.Series(input_df['isoform'].unique())
    isoform_numbers = list(isoforms.str[-1:].astype('int').unique())
    other_isoforms = []
    for isoform_number in isoform_numbers:
        if isoform_number==1:
            first_isoforms = list(isoforms.loc[isoforms.str.endswith('_1')])
        else:
            other_isoforms.append(list(isoforms.loc[isoforms.str.endswith('_'+str(isoform_number))]))
    junction_count_data[isoforms] = junction_count_data[isoforms].astype('float').fillna(0)
    junction_count_data['denom'] = junction_count_data[first_isoforms].mean(1)
    for oiso in other_isoforms:
        junction_count_data['denom'] = junction_count_data['denom']+junction_count_data[oiso].mean(1)
    junction_count_data['psi'] = (junction_count_data[first_isoforms].mean(1))/(junction_count_data['denom'])
    junction_count_data['t']=1
    df_gr = junction_count_data.groupby('smtsd').agg({'t':sum}).reset_index()
    junction_count_data = junction_count_data.loc[(~junction_count_data['psi'].isna())&(junction_count_data['denom']>15)]
    # порог на знаменатель - 15
    
    # если внутри ткани оказалось удалено более 80% образцов, то надо убрать такую ткань полностью
    df_gr_1 = junction_count_data.groupby('smtsd').agg({'t':sum}).reset_index().rename(columns={'t':'t1'})
    df_gr_1 = pd.merge(df_gr_1,df_gr,how='left',on='smtsd')
    df_gr_1['perc_deleted'] = ((df_gr_1['t']-df_gr_1['t1'])/df_gr_1['t']*100)
    df_gr_1['remove'] = ((df_gr_1['perc_deleted']>80)).astype('int')
    junction_count_data = pd.merge(junction_count_data,df_gr_1[['smtsd','remove']],how='left',on='smtsd')
    junction_count_data['remove'] = junction_count_data['remove'].fillna(0).astype('int')
    junction_count_data = junction_count_data.loc[junction_count_data['remove']==0]
    junction_count_data.drop(['t','remove'],1,inplace=True)
    # разделяем по квантилям psi
    thrL = junction_count_data['psi'].quantile(0.25)
    thrR = junction_count_data['psi'].quantile(0.75)
    psi_median = junction_count_data['psi'].median()
    junction_count_data['cluster'] = ((junction_count_data['psi']<=thrL).astype('int')*(-1)+(junction_count_data['psi']>=thrR).astype('int')).astype('str').str.replace('-1','LQ').replace('1','UQ').replace('0','MED')
    return junction_count_data

def add_gene_expression(junction_count_data,gene):
    # если в junction_count_data уже есть колонка с экспрессией гена, то ее не нужно снова добавлять
    if gene not in list(junction_count_data.columns):
        sql = """SELECT * FROM gene_counts_hg19 
                WHERE gene_name IN ("""+"'"+gene+"'"+""")
                """
        dat1 = pd.read_sql_query(sql, conn)
        reg = dat1['gene_counts'].astype('str').str[1:-1].str.split(', ',expand=True).transpose()
        reg.columns = list(dat1['gene_name'])
        reg = reg.astype('int')
        reg['sample_id'] = reg.index+1
        junction_count_data = pd.merge(junction_count_data,reg,how='left',on='sample_id')
        junction_count_data[gene] = np.log2((1/junction_count_data['sf_global'])*(junction_count_data[gene]+8))
        junction_count_data[gene] = junction_count_data[gene]-junction_count_data[gene].median()
    if 'local_expr' not in list(junction_count_data.columns):
        junction_count_data['local_expr'] = np.log2((1/junction_count_data['sf'])*(junction_count_data['denom']))
        junction_count_data['local_expr'] = junction_count_data['local_expr']-junction_count_data['local_expr'].median()
    return junction_count_data

def compare_psi_with_gene_expression_gtex(junction_count_data,gene,AS_type):
    junction_count_data = add_gene_expression(junction_count_data,gene)
    # разделяем данные на верхний и нижний кластер по psi, и сравниваем их. Используем модель для присвоения p-values
    a = junction_count_data.loc[junction_count_data['cluster']=='LQ'][[gene,'local_expr','psi']]
    b = junction_count_data.loc[junction_count_data['cluster']=='UQ'][[gene,'local_expr','psi']]
    if len(a)>0 and len(b)>0:
        stat = stats.mannwhitneyu(a[gene],b[gene],alternative='greater')[0]
        zscore_global = (stat-len(a)*len(b)/2)/(len(a)*len(b)*(len(a)+len(b)+1)/12)**0.5
        stat = stats.mannwhitneyu(a['local_expr'],b['local_expr'],alternative='greater')[0]
        zscore_local = (stat-len(a)*len(b)/2)/(len(a)*len(b)*(len(a)+len(b)+1)/12)**0.5
        psi_median = junction_count_data['psi'].median()
        # По-хорошему нужно для каждого типа событий построить свой контроль
        control_data = pd.read_csv('/home/magmir/TASS/NMD_regulation/control_events/control_CE.tsv',delimiter="\t",
                                       index_col=None,header=0)
        # выбираем маленькую окрестность по psi_median
        control_data = control_data.loc[abs(control_data['psi_median']-psi_median)<0.01]
        # тест для локальной экспрессии
        mu = control_data['local_mu_norm'].mean()
        stdev = control_data['local_stdev_norm'].mean()
        pval_local = 1-stats.norm.cdf(zscore_local,loc=mu,scale=stdev)
        # домножаем на (-1), т.к. тесты делались в ожидании отрицательной взаимосвязи, а на картинке и в отчете это не ожиданно
        log2FC_local = (a['local_expr'].median()-b['local_expr'].median())*(-1) 
        log2FC_local = np.round(log2FC_local,2)

        # тест для глобальной экспрессии
        mu = control_data['global_mu_norm'].mean()
        stdev = control_data['global_stdev_norm'].mean()
        pval_global = 1-stats.norm.cdf(zscore_global,loc=mu,scale=stdev)
        # домножаем на (-1), т.к. тесты делались в ожидании отрицательной взаимосвязи, а на картинке и в отчете это не ожиданно
        log2FC_global = (a[gene].median()-b[gene].median())*(-1)
        log2FC_global = np.round(log2FC_global,2)
        psi_L = np.round(a['psi'].median(),3)
        psi_U = np.round(b['psi'].median(),3)

        gtex_psi_vs_expr = pd.DataFrame([psi_median,psi_L,psi_U,zscore_local,log2FC_local,pval_local,zscore_global,log2FC_global,pval_global]).transpose()
        gtex_psi_vs_expr.columns = ['psi_median','psi_L','psi_U','zscore_local','log2FC_local',
                      'pval_local','zscore_global','log2FC_global','pval_global']
    else:
        gtex_psi_vs_expr = pd.DataFrame(columns = ['psi_median','psi_L','psi_U','zscore_local','log2FC_local',
                      'pval_local','zscore_global','log2FC_global','pval_global'])
    return [junction_count_data,gtex_psi_vs_expr]

def add_regulators_expression(junction_count_data,regulators):
    regulators_new = regulators.copy()
    for regulator in regulators:
        if regulator in list(junction_count_data.columns):
            regulators_new.remove(regulator)
    if len(regulators_new)>0:
        sql = """SELECT * FROM gene_counts_hg19 
                WHERE gene_name IN ("""+str(regulators_new)[1:-1]+""")
                """
        dat1 = pd.read_sql_query(sql, conn)
        reg = dat1['gene_counts'].astype('str').str[1:-1].str.split(', ',expand=True).transpose()
        reg.columns = list(dat1['gene_name'])
        reg = reg.astype('int')
        reg['sample_id'] = reg.index+1
        junction_count_data = pd.merge(junction_count_data,reg,how='left',on='sample_id')
        for regulator in regulators_new:
            if regulator in junction_count_data.columns:
                junction_count_data[regulator] = np.log2((1/junction_count_data['sf_global'])*(junction_count_data[regulator]+8))
                junction_count_data[regulator] = junction_count_data[regulator]-junction_count_data[regulator].median()
            else:
                junction_count_data[regulator] = 0
    return junction_count_data
    
def get_correlations_of_gene_expression_with_regulators_gtex(junction_count_data,gene,regulators,partial_cors=False,glasso=False,rscript_path='',glasso_reg_param=0.1):
    # экспрессия регуляторов против экспрессии гена
    # если в junction_count_data уже есть колонки с экспрессией регуляторов и гена, то их не нужно снова добавлять
    # если в junction_count_data уже есть колонка с экспрессией гена, то ее не нужно снова добавлять
    junction_count_data = add_regulators_expression(junction_count_data,regulators)
    junction_count_data = add_gene_expression(junction_count_data,gene)
    # разделяем данные на верхний и нижний кластер по глобальной экспрессии гена, и сравниваем их. Смотрим на частичные корреляции
    thrL = junction_count_data[gene].quantile(0.25)
    thrR = junction_count_data[gene].quantile(0.75)
    expr_median = junction_count_data[gene].median()
    junction_count_data['cluster_gene'] = ((junction_count_data[gene]<=thrL).astype('int')*(-1)+(junction_count_data[gene]>=thrR).astype('int')).astype('str').str.replace('-1','LQ').replace('1','UQ').replace('0','MED')

    log2FCs = []
    pvals_clust = []
    rhos = []
    pvals_rho = []
    if glasso==True:
        glasso_estimates = []
        pvals_glasso = []
    for regulator in regulators:
        pval_rho = stats.spearmanr(junction_count_data[regulator],junction_count_data[gene])[1]
        pvals_rho.append(min(pval_rho*len(regulators),1)) # делаем поправку bonferroni
        rho = np.round(stats.spearmanr(junction_count_data[regulator],junction_count_data[gene])[0],2)     
        rhos.append(rho)
        a = junction_count_data.loc[junction_count_data['cluster_gene']=='LQ'][regulator]
        b = junction_count_data.loc[junction_count_data['cluster_gene']=='UQ'][regulator]
        pval_clust = stats.mannwhitneyu(a,b)[1]
        pvals_clust.append(min(pval_clust*len(regulators),1)) # делаем поправку bonferroni
        log2FC = np.round(np.median(b)-np.median(a),2)
        log2FCs.append(log2FC)
    regulators_res = pd.DataFrame([regulators,log2FCs,pvals_clust,rhos,pvals_rho]).transpose()
    regulators_res.columns = ['regulator','log2FC_global','pval_clust_adj_global','rho_global','pval_rho_adj_global']
    print('global expression cluster comparison done')
    if partial_cors==True or glasso==True:
        junction_count_data[[gene]+regulators].to_csv(
            '/home/magmir/TASS/NMD_regulation/temp/global_gene_vs_regulators.tsv', 
                          sep=str('\t'),index=None,header=True,encoding='utf-8')
    if partial_cors==True:
        # save to calculate partial correlations in R
        os.system(rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_partial_correlations.r '+\
                  '/home/magmir/TASS/NMD_regulation/temp/global_gene_vs_regulators.tsv '+\
                 '/home/magmir/TASS/NMD_regulation/temp/pcor_global_expr '+\
                 '/home/magmir/libs/R '+\
                 'spearman')
        pcor_estimates = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/pcor_global_expr.estimate.tsv',
                     delimiter="\t",index_col=None,header=0,encoding = "utf-8")
        pcor_pvalues = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/pcor_global_expr.pvalue.tsv',
                     delimiter="\t",index_col=None,header=0,encoding = "utf-8")
        pcors = pcor_estimates.head(1)[pcor_estimates.columns[1:]].transpose()
        pcors.columns = ['pcor_global']
        pcors['regulator'] = pcors.index
        pvals_pcor = pcor_pvalues.head(1)[pcor_pvalues.columns[1:]].transpose()
        pvals_pcor.columns = ['pval_pcor_global']
        pvals_pcor['regulator'] = pvals_pcor.index
        pvals_pcor['pval_pcor_global'] = pvals_pcor['pval_pcor_global']*len(regulators) # делаем поправку bonferroni
        pvals_pcor['t']=1
        pvals_pcor['pval_pcor_adj_global'] = pvals_pcor[['pval_pcor_global','t']].min(1)
        regulators_res = pd.merge(regulators_res,pcors,how='left',on='regulator')
        regulators_res = pd.merge(regulators_res,pvals_pcor[['regulator','pval_pcor_adj_global']],how='left',on='regulator')
        print('global expression partial cors done')
    if glasso==True:        
        reg_param = glasso_reg_param
        os.system(rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_glasso.r '+\
              '/home/magmir/TASS/NMD_regulation/temp/global_gene_vs_regulators.tsv '+\
             '/home/magmir/TASS/NMD_regulation/temp/glasso_global_expr '+\
             '/home/magmir/libs/R '+\
             'spearman '+\
            str(reg_param))    
        glasso_estimates = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/glasso_global_expr.estimate.tsv',
                     delimiter="\t",index_col=None,header=0,encoding = "utf-8")
        glassos = glasso_estimates.head(1)[glasso_estimates.columns[1:]].transpose()
        glassos.columns = ['glasso_pcor_global']
        glassos['regulator'] = glassos.index
        regulators_res = pd.merge(regulators_res,glassos,how='left',on='regulator')
        print('global expression glasso done')
    # разделяем данные на верхний и нижний кластер по локальной экспрессии гена, и сравниваем их. Смотрим на частичные корреляции
    thrL = junction_count_data['local_expr'].quantile(0.25)
    thrR = junction_count_data['local_expr'].quantile(0.75)
    expr_median = junction_count_data['local_expr'].median()
    junction_count_data['cluster_gene'] = ((junction_count_data['local_expr']<=thrL).astype('int')*(-1)+(junction_count_data['local_expr']>=thrR).astype('int')).astype('str').str.replace('-1','LQ').replace('1','UQ').replace('0','MED')

    log2FCs = []
    pvals_clust = []
    rhos = []
    pvals_rho = []
    if glasso==True:
        glasso_estimates = []
        pvals_glasso = []
    for regulator in regulators:
        pval_rho = stats.spearmanr(junction_count_data[regulator],junction_count_data['local_expr'])[1]
        pvals_rho.append(min(pval_rho*len(regulators),1)) # делаем поправку bonferroni
        rho = np.round(stats.spearmanr(junction_count_data[regulator],junction_count_data['local_expr'])[0],2)     
        rhos.append(rho)
        a = junction_count_data.loc[junction_count_data['cluster_gene']=='LQ'][regulator]
        b = junction_count_data.loc[junction_count_data['cluster_gene']=='UQ'][regulator]
        pval_clust = stats.mannwhitneyu(a,b)[1]
        pvals_clust.append(min(pval_clust*len(regulators),1)) # делаем поправку bonferroni
        log2FC = np.round(np.median(b)-np.median(a),2)
        log2FCs.append(log2FC)
    regulators_res_local = pd.DataFrame([regulators,log2FCs,pvals_clust,rhos,pvals_rho]).transpose()
    regulators_res_local.columns = ['regulator','log2FC_local','pval_clust_adj_local','rho_local','pval_rho_adj_local']
    print('local expression cluster comparison done')
    if partial_cors==True or glasso==True:
        junction_count_data[['local_expr']+regulators].to_csv(
            '/home/magmir/TASS/NMD_regulation/temp/local_gene_vs_regulators.tsv', 
                          sep=str('\t'),index=None,header=True,encoding='utf-8')
    if partial_cors==True:
        # save to calculate partial correlations in R
        os.system(rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_partial_correlations.r '+\
                  '/home/magmir/TASS/NMD_regulation/temp/local_gene_vs_regulators.tsv '+\
                 '/home/magmir/TASS/NMD_regulation/temp/pcor_local_expr '+\
                 '/home/magmir/libs/R '+\
                 'spearman')
        pcor_estimates = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/pcor_local_expr.estimate.tsv',
                     delimiter="\t",index_col=None,header=0,encoding = "utf-8")
        pcor_pvalues = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/pcor_local_expr.pvalue.tsv',
                     delimiter="\t",index_col=None,header=0,encoding = "utf-8")
        pcors = pcor_estimates.head(1)[pcor_estimates.columns[1:]].transpose()
        pcors.columns = ['pcor_local']
        pcors['regulator'] = pcors.index
        pvals_pcor = pcor_pvalues.head(1)[pcor_pvalues.columns[1:]].transpose()
        pvals_pcor.columns = ['pval_pcor_local']
        pvals_pcor['regulator'] = pvals_pcor.index
        pvals_pcor['pval_pcor_local'] = pvals_pcor['pval_pcor_local']*len(regulators) # делаем поправку bonferroni
        pvals_pcor['t']=1
        pvals_pcor['pval_pcor_adj_local'] = pvals_pcor[['pval_pcor_local','t']].min(1)
        regulators_res_local = pd.merge(regulators_res_local,pcors,how='left',on='regulator')
        regulators_res_local = pd.merge(regulators_res_local,pvals_pcor[['regulator','pval_pcor_adj_local']],how='left',on='regulator')
        print('local expression partial cors done')
    if glasso==True:        
        reg_param = glasso_reg_param
        os.system(rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_glasso.r '+\
              '/home/magmir/TASS/NMD_regulation/temp/local_gene_vs_regulators.tsv '+\
             '/home/magmir/TASS/NMD_regulation/temp/glasso_local_expr '+\
             '/home/magmir/libs/R '+\
             'spearman '+\
            str(reg_param))    
        glasso_estimates = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/glasso_local_expr.estimate.tsv',
                     delimiter="\t",index_col=None,header=0,encoding = "utf-8")
        glassos = glasso_estimates.head(1)[glasso_estimates.columns[1:]].transpose()
        glassos.columns = ['glasso_pcor_local']
        glassos['regulator'] = glassos.index
        regulators_res_local = pd.merge(regulators_res_local,glassos,how='left',on='regulator')
        print('local expression glasso done')
    regulators_res = pd.merge(regulators_res,regulators_res_local,how='left',on='regulator')
    return [junction_count_data,regulators_res]

def get_correlations_of_psi_with_regulators_gtex(junction_count_data,regulators,partial_cors=False,glasso=False,backward_association=False,rscript_path='',glasso_reg_param=10):
    # экспрессия регуляторов
    # если в junction_count_data уже есть колонки с экспрессией регуляторов, то их не нужно снова добавлять
    junction_count_data = add_regulators_expression(junction_count_data,regulators)
    # разделяем данные на верхний и нижний кластер, и сравниваем их. Смотрим на частичные корреляции
    log2FCs = []
    pvals_clust = []
    rhos = []
    pvals_rho = []
    if glasso==True:
        glasso_estimates = []
        pvals_glasso = []
    for regulator in regulators:
        pval_rho = stats.spearmanr(junction_count_data[regulator],junction_count_data['psi'])[1]
        pvals_rho.append(min(pval_rho*len(regulators),1)) # делаем поправку bonferroni
        rho = np.round(stats.spearmanr(junction_count_data[regulator],junction_count_data['psi'])[0],2)     
        rhos.append(rho)
        a = junction_count_data.loc[junction_count_data['cluster']=='LQ'][regulator]
        b = junction_count_data.loc[junction_count_data['cluster']=='UQ'][regulator]
        pval_clust = stats.mannwhitneyu(a,b)[1]
        pvals_clust.append(min(pval_clust*len(regulators),1)) # делаем поправку bonferroni
        log2FC = np.round(np.median(b)-np.median(a),2)
        log2FCs.append(log2FC)
    regulators_res = pd.DataFrame([regulators,log2FCs,pvals_clust,rhos,pvals_rho]).transpose()
    regulators_res.columns = ['regulator','log2FC','pval_clust_adj','rho','pval_rho_adj']
    print('cluster comparison done')
    if backward_association==True:
        back_delta_psi = []
        back_pvals_clust = []
        back_log2FCs = []
        for regulator in regulators:
            thrL = junction_count_data[regulator].quantile(0.25)
            thrR = junction_count_data[regulator].quantile(0.75)
            junction_count_data['cluster_'+regulator] = ((junction_count_data[regulator]<=thrL).astype('int')*(-1)+(junction_count_data[regulator]>=thrR).astype('int')).astype('str').str.replace('-1','LQ').replace('1','UQ').replace('0','MED')
            a = junction_count_data.loc[junction_count_data['cluster_'+regulator]=='LQ']
            b = junction_count_data.loc[junction_count_data['cluster_'+regulator]=='UQ']
            pval_clust = stats.mannwhitneyu(a['psi'],b['psi'])[1]
            back_pvals_clust.append(min(pval_clust*len(regulators),1)) # делаем поправку bonferroni
            delta_psi = np.round(np.median(b['psi'])-np.median(a['psi']),2)
            back_delta_psi.append(delta_psi)
            back_log2FC = np.round(np.median(b[regulator])-np.median(a[regulator]),2)
            back_log2FCs.append(back_log2FC)
        regulators_res['back_delta_psi'] = back_delta_psi
        regulators_res['back_pval_clust_adj'] = back_pvals_clust
        regulators_res['back_log2FC'] = back_log2FCs
        
    if partial_cors==True or glasso==True:
        junction_count_data[['psi']+regulators].to_csv(
            '/home/magmir/TASS/NMD_regulation/temp/psi_vs_regulators.tsv', 
                          sep=str('\t'),index=None,header=True,encoding='utf-8')
    if partial_cors==True:
        # save to calculate partial correlations in R
        os.system(rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_partial_correlations.r '+\
                  '/home/magmir/TASS/NMD_regulation/temp/psi_vs_regulators.tsv '+\
                 '/home/magmir/TASS/NMD_regulation/temp/pcor '+\
                 '/home/magmir/libs/R '+\
                 'spearman')
        pcor_estimates = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/pcor.estimate.tsv',
                     delimiter="\t",index_col=None,header=0,encoding = "utf-8")
        pcor_pvalues = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/pcor.pvalue.tsv',
                     delimiter="\t",index_col=None,header=0,encoding = "utf-8")
        pcors = pcor_estimates.head(1)[pcor_estimates.columns[1:]].transpose()
        pcors.columns = ['pcor']
        pcors['regulator'] = pcors.index
        pvals_pcor = pcor_pvalues.head(1)[pcor_pvalues.columns[1:]].transpose()
        pvals_pcor.columns = ['pval_pcor']
        pvals_pcor['regulator'] = pvals_pcor.index
        pvals_pcor['pval_pcor'] = pvals_pcor['pval_pcor']*len(regulators) # делаем поправку bonferroni
        pvals_pcor['t']=1
        pvals_pcor['pval_pcor_adj'] = pvals_pcor[['pval_pcor','t']].min(1)
        regulators_res = pd.merge(regulators_res,pcors,how='left',on='regulator')
        regulators_res = pd.merge(regulators_res,pvals_pcor[['regulator','pval_pcor_adj']],how='left',on='regulator')
        print('partial cors done')
    if glasso==True:        
        reg_param = len(junction_count_data)*glasso_reg_param
        os.system(rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_glasso.r '+\
              '/home/magmir/TASS/NMD_regulation/temp/psi_vs_regulators.tsv '+\
             '/home/magmir/TASS/NMD_regulation/temp/glasso '+\
             '/home/magmir/libs/R '+\
             'spearman '+\
            str(reg_param))    
        glasso_estimates = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/glasso.estimate.tsv',
                     delimiter="\t",index_col=None,header=0,encoding = "utf-8")
        glassos = glasso_estimates.head(1)[glasso_estimates.columns[1:]].transpose()
        glassos.columns = ['glasso_pcor']
        glassos['regulator'] = glassos.index
        regulators_res = pd.merge(regulators_res,glassos,how='left',on='regulator')
        print('glasso done')
    return [junction_count_data,regulators_res]

def tissue_specificity_of_psi_and_gene_expression(junction_count_data,gene_name,rscript_path,tissue_field):
    return tissue_specificity_of_psi_and_regulator_expression(junction_count_data,[gene_name],rscript_path,tissue_field)

def tissue_specificity_of_psi_and_regulator_expression(junction_count_data,regulators,rscript_path,tissue_field):
    # добавляем информацию о регуляторах, если нужно
    junction_count_data = add_regulators_expression(junction_count_data,regulators)
    
    # тестируем все ткани и пытаемся найти такие, в которых psi значительно отличается от остальных
    tissue_df = []
    for tissue in list(junction_count_data[tissue_field].unique()):
        a = junction_count_data.loc[junction_count_data[tissue_field]!=tissue]
        b = junction_count_data.loc[junction_count_data[tissue_field]==tissue]
        if len(a)==0 or len(b)==0:
            continue
        delta_psi_pval = stats.mannwhitneyu(a['psi'],b['psi'])[1]
        delta_psi = b['psi'].median()-a['psi'].median()
        reg_res=[]
        reg_header = []
        for regulator in regulators:
            pval = stats.mannwhitneyu(a[regulator],b[regulator])[1]
#             pval = min(pval*len(regulators),1) # делаем поправку бонферрони, т.к. мы будем искать хотя бы один регулятор
            log2FC = b[regulator].median()-a[regulator].median()
            reg_res = reg_res+[log2FC,pval]
            reg_header = reg_header+['log2FC_'+regulator,'pval_'+regulator]
        tissue_df.append([tissue,delta_psi,delta_psi_pval]+reg_res)
    if tissue_df==[]:
        return [junction_count_data,None]
    tissue_df = pd.DataFrame(tissue_df,columns=[tissue_field,'delta_psi','delta_psi_pval']+reg_header)
    tissue_df['id'] =tissue_df.index
    tissue_df[['id','delta_psi_pval']].to_csv('/home/magmir/TASS/NMD_regulation/temp/ts_expr_pvals.tsv', 
                          sep=str('\t'),index=None,header=True,encoding='utf-8')
    if os.path.isfile('/home/magmir/TASS/NMD_regulation/temp/ts_expr_qvals.tsv'):
        os.system('rm /home/magmir/TASS/NMD_regulation/temp/ts_expr_qvals.tsv')
    os.system(rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_q_values.r '+\
              '/home/magmir/TASS/NMD_regulation/temp/ts_expr_pvals.tsv '+\
             '/home/magmir/TASS/NMD_regulation/temp/ts_expr_qvals.tsv '+\
             '/home/magmir/libs/R ')
    result_qvals = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/ts_expr_qvals.tsv',
                                        delimiter="\t",index_col=None,header=0)
    result_qvals.rename(columns={'qval':'delta_psi_qval'},inplace=True)
    tissue_df = pd.merge(tissue_df,result_qvals.drop(['pval','lFDR'],1),how='left',on='id')
    for regulator in regulators:
        tissue_df[['id','pval_'+regulator]].to_csv(
            '/home/magmir/TASS/NMD_regulation/temp/ts_expr_pvals.tsv', 
                          sep=str('\t'),index=None,header=True,encoding='utf-8')
        if os.path.isfile('/home/magmir/TASS/NMD_regulation/temp/ts_expr_qvals.tsv'):
            os.system('rm /home/magmir/TASS/NMD_regulation/temp/ts_expr_qvals.tsv')
        os.system(rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_q_values.r '+\
              '/home/magmir/TASS/NMD_regulation/temp/ts_expr_pvals.tsv '+\
             '/home/magmir/TASS/NMD_regulation/temp/ts_expr_qvals.tsv '+\
             '/home/magmir/libs/R ')
        result_qvals = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/ts_expr_qvals.tsv',
                                        delimiter="\t",index_col=None,header=0)
        result_qvals.rename(columns={'qval':'qval_'+regulator},inplace=True)
        tissue_df = pd.merge(tissue_df,result_qvals.drop(['pval','lFDR'],1),how='left',on='id')
    tissue_df.drop('id',1,inplace=True)
    return [junction_count_data,tissue_df]        

# перевод координат из ipsa в rMATS
def get_rMATS_coordinates(input_df,AS_type):
    
    if AS_type not in ['CE','alt 5SS','alt 3SS','MXE','IR']:
        return None
    
    tmp = input_df['long_id'].str.split('_',expand=True)
    l = len(tmp.columns)
    input_df['chr'] = tmp[0]
    input_df['start'] = tmp[1].astype('int')
    input_df['str'] = input_df.apply(lambda x: x['long_id'].split('_')[3] if 'junction' in x['isoform'] else x['long_id'].split('_')[2],1)
    input_df['end'] = input_df.apply(lambda x: x['long_id'].split('_')[2] if 'junction' in x['isoform'] else 0,1)
    input_df['end'] = input_df['end'].astype('int')
    
    if len(input_df['chr'].unique())>1 or len(input_df['str'].unique())>1:
        print('Incorrect specification of chr or str in input_df')
        return None
    else:
        chrom = input_df['chr'].iloc[0]
        strand = input_df['str'].iloc[0]
    
    starts = list(input_df['start'].unique())
    starts.sort()
    
    ends = list(input_df.loc[input_df['end']!=0]['end'].unique())
    ends.sort()
    
    if AS_type=='CE':
        if not (len(starts)==2 and len(ends)==2):
            print('Incorrect specification of junctions submitted for '+AS_type+' event')
            return None
        
        upstreamEE = int(starts[0])
        downstreamES = int(ends[1])-1
        exonStart_0base = int(ends[0])-1
        exonEnd = int(starts[1])
        
        skip_isoform = chrom+'_'+str(upstreamEE)+'_'+str(downstreamES+1)+'_'+strand
        
        rmats_file = 'SE.MATS.JC'
        
        number = list(input_df.loc[input_df['long_id'].str.contains(skip_isoform)]['number'].unique())
        if len(number)>1:
            print('Incorrect specification of the numbers for isoforms')
            return None
        else:
            if number[0]==1:
                adj_coef = -1
            else:
                adj_coef = 1
            
        res = pd.DataFrame([['chr','strand','upstreamee','downstreames','exonstart_0base','exonend','rmats_file','adj_coef'],
                            [chrom,strand,upstreamEE,downstreamES,exonStart_0base,exonEnd,rmats_file,adj_coef]]).transpose()
        res.columns=['param','val']
        return res
    
    elif AS_type=='alt 5SS':
        rmats_file = 'A5SS.MATS.JC'
        
        if strand=='+':
            if not (len(starts)==2 and len(ends)==1):
                print('Incorrect specification of junctions submitted for '+AS_type+' event')
                return None
            
            flankingES = int(ends[0])-1
            shortEE = int(starts[0])
            longExonEnd = int(starts[1])
            
            long_isoform = chrom+'_'+str(longExonEnd)+'_'+str(flankingES+1)+'_'+strand
            number = list(input_df.loc[input_df['long_id'].str.contains(long_isoform)]['number'].unique())
            
            if len(number)>1:
                print('Incorrect specification of the numbers for isoforms')
                return None
            else:
                if number[0]==1:
                    adj_coef = 1
                else:
                    adj_coef = -1
                    
            res = pd.DataFrame([['chr','strand','flankinges','shortee','longexonend','rmats_file','adj_coef'],
                                [chrom,strand,flankingES,shortEE,longExonEnd,rmats_file,adj_coef]]).transpose()
            res.columns=['param','val']
            return res 
        else:
            if not (len(starts)==1 and len(ends)==2):
                print('Incorrect specification of junctions submitted for '+AS_type+' event')
                return None
            flankingEE = int(starts[0])
            shortES = int(ends[1])-1
            longExonStart_0base = int(ends[0])-1
            
            long_isoform = chrom+'_'+str(flankingEE)+'_'+str(longExonStart_0base+1)+'_'+strand
            number = list(input_df.loc[input_df['long_id'].str.contains(long_isoform)]['number'].unique())
            
            if len(number)>1:
                print('Incorrect specification of the numbers for isoforms')
                return None
            else:
                if number[0]==1:
                    adj_coef = 1
                else:
                    adj_coef = -1

            res = pd.DataFrame([['chr','strand','flankingee','shortes','longexonstart_0base','rmats_file','adj_coef'],
                                [chrom,strand,flankingEE,shortES,longExonStart_0base,rmats_file,adj_coef]]).transpose()
            
            res.columns=['param','val']
            return res
        
    elif AS_type=='alt 3SS':
        rmats_file = 'A3SS.MATS.JC'
        if strand=='+':
            if not (len(starts)==1 and len(ends)==2):
                print('Incorrect specification of junctions submitted for '+AS_type+' event')
                return None
            
            flankingEE = int(starts[0])
            shortES = int(ends[1])-1
            longExonStart_0base = int(ends[0])-1
            
            long_isoform = chrom+'_'+str(flankingEE)+'_'+str(longExonStart_0base+1)+'_'+strand
            number = list(input_df.loc[input_df['long_id'].str.contains(long_isoform)]['number'].unique())
            
            if len(number)>1:
                print('Incorrect specification of the numbers for isoforms')
                return None
            else:
                if number[0]==1:
                    adj_coef = 1
                else:
                    adj_coef = -1

            res = pd.DataFrame([['chr','strand','flankingee','shortes','longexonstart_0base','rmats_file','adj_coef'],
                                [chrom,strand,flankingEE,shortES,longExonStart_0base,rmats_file,adj_coef]]).transpose()
            res.columns=['param','val']
            return res 
        else:
            if not (len(starts)==2 and len(ends)==1):
                print('Incorrect specification of junctions submitted for '+AS_type+' event')
                return None
            
            flankingES = int(ends[0])-1
            shortEE = int(starts[0])
            longExonEnd = int(starts[1])
            
            long_isoform = chrom+'_'+str(longExonEnd)+'_'+str(flankingES+1)+'_'+strand
            number = list(input_df.loc[input_df['long_id'].str.contains(long_isoform)]['number'].unique())
            
            if len(number)>1:
                print('Incorrect specification of the numbers for isoforms')
                return None
            else:
                if number[0]==1:
                    adj_coef = 1
                else:
                    adj_coef = -1
                    
            res = pd.DataFrame([['chr','strand','flankinges','shortee','longexonend','rmats_file','adj_coef'],
                                [chrom,strand,flankingES,shortEE,longExonEnd,rmats_file,adj_coef]]).transpose()
            res.columns=['param','val']
            return res
        
    elif AS_type=='MXE':       
        rmats_file = 'MXE.MATS.JC'
        if len(input_df)!=4:
            print('Incorrect number of junctions submitted for '+AS_type+' event')
            return None

        if strand=='+':
            if not (len(starts)==3 and len(ends)==3):
                print('Incorrect specification of junctions submitted for '+AS_type+' event')
            
            upstreamEE = int(starts[0])
            downstreamES = int(ends[2])-1
            
            firstExonStart_0base = int(ends[0])-1
            firstExonEnd = int(starts[1])
            
            secondExonStart_0base = int(ends[1])-1
            secondExonEnd = int(starts[2])
            
            first_exon_isoform_up = chrom+'_'+str(upstreamEE)+'_'+str(firstExonStart_0base+1)+'_'+strand
            first_exon_isoform_down = chrom+'_'+str(firstExonEnd)+'_'+str(downstreamES+1)+'_'+strand
            
            number = list(input_df.loc[input_df['long_id'].isin([first_exon_isoform_up,first_exon_isoform_down])]['number'].unique())
            
            if len(number)>1:
                print('Incorrect specification of the numbers for isoforms')
                return None
            else:
                if number[0]==1:
                    adj_coef = 1
                else:
                    adj_coef = -1

            res = pd.DataFrame([['chr','strand','upstreamee','downstreames',
                                 'firstexonstart_0base','firstexonend','secondexonstart_0base','secondexonend',
                                 'rmats_file','adj_coef'],[chrom,strand,upstreamEE,downstreamES,firstExonStart_0base,
                                                           firstExonEnd,secondExonStart_0base,
                                                           secondExonEnd,rmats_file,adj_coef]]).transpose()
            res.columns=['param','val']
            return res
        else:
            if not (len(starts)==3 and len(ends)==3):
                print('Incorrect specification ')
            
            upstreamEE = int(starts[0])
            downstreamES = int(ends[2])-1
            
            firstExonStart_0base = int(ends[0])-1
            firstExonEnd = int(starts[1])
            
            secondExonStart_0base = int(ends[1])-1
            secondExonEnd = int(starts[2])
            
            second_exon_isoform_up = chrom+'_'+str(upstreamEE)+'_'+str(secondExonStart_0base+1)+'_'+strand
            second_exon_isoform_down = chrom+'_'+str(secondExonEnd)+'_'+str(downstreamES+1)+'_'+strand
            
            number = list(input_df.loc[input_df['long_id'].isin([second_exon_isoform_up,second_exon_isoform_down])]['number'].unique())
            
            if len(number)>1:
                print('Incorrect specification of the numbers for isoforms')
                return None
            else:
                if number[0]==1:
                    adj_coef = 1
                else:
                    adj_coef = -1

            res = pd.DataFrame([['chr','strand','upstreamee','downstreames',
                                 'firstexonstart_0base','firstexonend','secondexonstart_0base','secondexonend',
                                 'rmats_file','adj_coef'],[chrom,strand,upstreamEE,downstreamES,firstExonStart_0base,
                                                           firstExonEnd,secondExonStart_0base,
                                                           secondExonEnd,rmats_file,adj_coef]]).transpose()
            res.columns=['param','val']
            return res
    elif AS_type=='IR':
        rmats_file = 'RI.MATS.JC'
        if strand=='+':
            if not (len(starts)==2 and len(ends)==1):
                print('Incorrect specification of junctions submitted for '+AS_type+' event')
                return None
            upstreamEE = int(starts[0])
            downstreamES = int(ends[0])-1

            excised_isoform = chrom+'_'+str(upstreamEE)+'_'+str(downstreamES+1)+'_'+strand
        
            number = list(input_df.loc[input_df['long_id']==excised_isoform]['number'].unique())

            if len(number)>1:
                print('Incorrect specification of the numbers for isoforms')
                return None
            else:
                if number[0]==1:
                    adj_coef = -1
                else:
                    adj_coef = 1
        else:
            if not (len(starts)==2 and len(ends)==1):
                print('Incorrect specification of junctions submitted for '+AS_type+' event')
                return None
            upstreamEE = int(starts[0])
            downstreamES = int(ends[0])-1

            excised_isoform = chrom+'_'+str(upstreamEE)+'_'+str(downstreamES+1)+'_'+strand
        
            number = list(input_df.loc[input_df['long_id']==excised_isoform]['number'].unique())

            if len(number)>1:
                print('Incorrect specification of the numbers for isoforms')
                return None
            else:
                if number[0]==1:
                    adj_coef = -1
                else:
                    adj_coef = 1
            
        res = pd.DataFrame([['chr','strand','upstreamee','downstreames','rmats_file','adj_coef'],
                            [chrom,strand,upstreamEE,downstreamES,rmats_file,adj_coef]]).transpose()
        res.columns=['param','val']
        return res

def get_sql_query_for_rmats(rMATS_coordinates):
    sql = ''
    rmats_file = rMATS_coordinates.loc[rMATS_coordinates['param']=='rmats_file']['val'].iloc[0]
    table_name = rmats_file.replace('.','_').lower()
    sql = sql+"""SELECT * FROM """+table_name
    sql = sql+""" WHERE """
    i=0
    for index, row in rMATS_coordinates.loc[~rMATS_coordinates['param'].isin(['rmats_file','adj_coef'])].iterrows():
        if i>0:
            sql = sql+""" AND """
        if row['param'] in ['chr','strand']:
            sql = sql+row['param']+""" = '"""+str(row['val'])+"""'"""
        else:
            sql = sql+row['param']+""" = """+str(row['val'])
        i=i+1
    return sql
    
def get_rMATS_data_from_NMD_inactivation(input_df,AS_type):
    res = get_rMATS_data_from_regulators_experiments(input_df,AS_type,SRA_directory,RBP_shRNA_KD_dir,rscript_path,regulators = ['UPF1','CHX'])
    return res

# смотрим на исследуемое событие AS в данных по нокдаунам, нокаутам и оверэкспрессии RBP
def get_rMATS_data_from_regulators_experiments(input_df,AS_type,regulators = None,exclude=None):
    rMATS_coordinates = get_rMATS_coordinates(input_df,AS_type)
    if rMATS_coordinates is None or len(rMATS_coordinates)<=2:
        print('rMATS coordinates were not defined')
        return None
    rmats_file = rMATS_coordinates.loc[rMATS_coordinates['param']=='rmats_file']
    if len(rmats_file)>0: 
        rmats_file = rmats_file.iloc[0]['val']
    else:
        print('rMATS file name not defined')
        return None
    adj_coef = rMATS_coordinates.loc[rMATS_coordinates['param']=='adj_coef']
    if len(adj_coef)>0: 
        adj_coef = adj_coef.iloc[0]['val']
    else:
        print('adj coefficient not defined')
        return None
    # строим query для запроса в дата фреймах
    query = ''
    for index, row in rMATS_coordinates.loc[~rMATS_coordinates['param'].isin(['rmats_file','adj_coef'])].iterrows():
        if query!='':
            query = query+' & '
        if row['param'] in ['strand','chr']:
            query = query+'@df_reg["'+row['param']+'"]'+'=="'+str(row['val'])+'"'
        else:
            query = query+'@df_reg["'+row['param']+'"]'+'=='+str(row['val'])
    # данные ENCODE по нокдаунам RBP
    shRNA_table = pd.read_csv('/home/magmir/TASS/RBP_shRNA_KD/shRNA_table.tsv',delimiter="\t",index_col=None,header=None)
    shRNA_table.columns = ['KD1','KD2','CTL1','CTL2','cell_line','target','tmp']
    shRNA_table.drop('tmp',1,inplace=True)
    # убираем эксперименты с неудавшимся нокдауном
    shRNA_table = shRNA_table.loc[~((shRNA_table['target']=='UPF1')&(shRNA_table['cell_line']=='HepG2'))]
    # дополнительные данные из sra
    public_data_on_AS_NMD = pd.read_csv('/home/magmir/TASS/NMD_regulation/public_data_on_AS_NMD.txt',delimiter="\t",index_col=None,header=0)    
    # по умолчанию мы не знаем, какие регуляторы у события, и нам надо проверить все возможные варианты. 
    if regulators is None or regulators==[]:
        public_data_on_AS_NMD_RBP_list = pd.read_csv('/home/magmir/TASS/NMD_regulation/public_data_on_AS_NMD_RBP_list.txt',delimiter="\t",index_col=None,header=None)
        regulators = list(np.unique(list(public_data_on_AS_NMD_RBP_list[0].unique())+list(shRNA_table['target'])))
        regulators.remove('UPF1') # на ингибирование NMD будем смотреть отдельно
    if exclude is not None:
        for regulator in exclude:
            if regulator in regulators:
                regulators.remove(regulator)
    result = []
    print('begin multiprocessing job.')
    with Manager() as manager:
        result = manager.list()
        p = Pool(multiprocessing.cpu_count()-5)
        for regulator in regulators:
            p.apply_async(get_rMATS_data_for_regulator, args=(result,regulator,shRNA_table,rmats_file,query,adj_coef,
                                                        RBP_shRNA_KD_dir,public_data_on_AS_NMD,SRA_directory))
        p.close()
        p.join()
        result = list(result)
    result = pd.DataFrame(result,columns=['regulator','Database','cell_line','experiment','pval','delta_psi'])
    if len(result)>0:
        result['id'] = result.index
        result[['id','pval']].to_csv(
                '/home/magmir/TASS/NMD_regulation/temp/RBP_experiments_pvals.tsv', 
                              sep=str('\t'),index=None,header=True,encoding='utf-8')
        if os.path.isfile('/home/magmir/TASS/NMD_regulation/temp/RBP_experiments_qvals.tsv'):
            os.system('rm /home/magmir/TASS/NMD_regulation/temp/RBP_experiments_qvals.tsv')
        os.system(rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_q_values.r '+\
                  '/home/magmir/TASS/NMD_regulation/temp/RBP_experiments_pvals.tsv '+\
                 '/home/magmir/TASS/NMD_regulation/temp/RBP_experiments_qvals.tsv '+\
                 '/home/magmir/libs/R ')
        result_qvals = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/RBP_experiments_qvals.tsv',
                                            delimiter="\t",index_col=None,header=0)
        result = pd.merge(result,result_qvals.drop('pval',1),how='left',on='id')
        result.drop('id',1,inplace=True)
    else:
        result = pd.DataFrame(columns=['regulator','Database','cell_line','experiment','pval','delta_psi','qval','lFDR'])
    return result    

def get_rMATS_data_for_regulator(result,regulator,shRNA_table,rmats_file,query,adj_coef,RBP_shRNA_KD_dir,public_data_on_AS_NMD,SRA_directory):
    # сначала ищем в данных ENCODE
    shRNA_df = shRNA_table.loc[shRNA_table['target']==regulator]
    if len(shRNA_df)>0:
        for index, row in shRNA_df.iterrows():
            rmats_file_path = RBP_shRNA_KD_dir+regulator+'/'+row['cell_line']+'/rmats_output/'+rmats_file+'.gz'
            deseq_file_path = RBP_shRNA_KD_dir+regulator+'/'+row['cell_line']+'/deseq2/res.tsv.gz'
            if os.path.isfile(rmats_file_path) and os.path.isfile(deseq_file_path):
                df_reg = pd.read_csv(rmats_file_path,delimiter="\t",index_col=None,header=0,compression='gzip')
                # проверяем, что есть эффект нокдауна. Если ошибочно эффект в другую сторону - меняем знак
                deseq = pd.read_csv(deseq_file_path,delimiter="\t",index_col=None,header=0,compression='gzip')
                x = deseq.loc[deseq['gene_name']==regulator].iloc[0]
                if (x['log2FoldChange']>0.5):
                    coef = -1
                elif (abs(x['log2FoldChange'])<=0.5):
                    result.append([regulator,'ENCODE',row['cell_line'],regulator+' shRNA KD (not efficient)',1,0])
                    continue
                    # неудачный нокдаун
                else:
                    coef = 1
                df_reg = df_reg.query(query)
                if len(df_reg)>0:
                    # нам нужен не FDR, а p-value от rMATS, которое мы потом скорректируем сами
                    PValue = df_reg.iloc[0]['PValue']
                    # обязательно нужна корректировка (-1)*..., т.к. неверно был дан инпут в rMATS для ENCODE shRNA RBP 
                    delta_psi = (-1)*df_reg.iloc[0]['IncLevelDifference']*adj_coef*coef
                    result.append([regulator,'ENCODE',row['cell_line'],regulator+' shRNA KD',PValue,delta_psi])
                else:
                    result.append([regulator,'ENCODE',row['cell_line'],regulator+' shRNA KD',1,0])
    # дополнительно смотрим в sra
    public_data_on_AS_NMD_df = public_data_on_AS_NMD.loc[public_data_on_AS_NMD['Experiments'].str.contains(regulator)]
    if len(public_data_on_AS_NMD_df)>0:
        for index, row in public_data_on_AS_NMD_df.iterrows():
            rmats_file_path = SRA_directory+row['SRP']+'/rMATS/hg19/'+('' if pd.isna(row['sub_path']) else row['sub_path'])+'rmats_output/'+rmats_file+'.gz'
            deseq_file_path = SRA_directory+row['SRP']+'/deseq2/hg19/'+('' if pd.isna(row['sub_path']) else row['sub_path'])+'res.tsv.gz'
            if os.path.isfile(rmats_file_path) and (os.path.isfile(deseq_file_path) or regulator=='CHX' or row['Experiments']==' UPF1 siRNA + XRN1 siRNA'):
                df_reg = pd.read_csv(rmats_file_path,delimiter="\t",index_col=None,header=0,compression='gzip')
                if regulator=='CHX' or row['Experiments']==' UPF1 siRNA + XRN1 siRNA' or ('deletion mutant' in row['Experiments']) or ('KO' in row['Experiments']):
                    coef=1
                else:
                    deseq = pd.read_csv(deseq_file_path,delimiter="\t",index_col=None,header=0,compression='gzip')
                    # проверяем, что эффект от нокдауна или оверэкспрессии в нужную сторону
                    x = deseq.loc[deseq['gene_name']==regulator].iloc[0]
                    if (regulator+' OE' in row['Experiments']):
                        expected_log2FC = 1
                    elif (regulator+' KD' in row['Experiments']) or (regulator+' siRNA' in row['Experiments']) or (regulator+' shRNA' in row['Experiments']):
                        expected_log2FC = -1
                    else:
                        expected_log2FC = 0
                    if (x['log2FoldChange']*expected_log2FC<-0.5):
                        coef = -1
                    elif (abs(x['log2FoldChange'])<=0.5):
                        # неудавшийся эксперимент
                        result.append([regulator,row['SRP'],row['Cell lines'],row['Experiments']+' (not efficient)',1,0])
                        continue
                    else:
                        coef = 1
                df_reg = df_reg.query(query)
                if len(df_reg)>0:
                    numb_of_samples = len(row['Annotation'].split(' '))
                    # если есть только по 1 образцу на условие, то принудительно ставим Pvalue=1, так же как в Deseq
                    if numb_of_samples==2:
                        PValue=1
                    else:
                        PValue = df_reg.iloc[0]['PValue']
                    delta_psi = df_reg.iloc[0]['IncLevelDifference']*adj_coef*row['corr_coef']*coef
                    result.append([regulator,row['SRP'],row['Cell lines'],row['Experiments'],PValue,delta_psi])
                else:
                    result.append([regulator,row['SRP'],row['Cell lines'],row['Experiments'],1,0])

def get_corr_field_value(x,field):
    if x['regulator']+' OE' in x['experiment']:
        return x[field]*(-1)
    else:
        return x[field]
                    
def group_experiments_results(experiments_on_regulators,field):                    
    experiments_on_regulators['abs_'+field] = abs(experiments_on_regulators[field])
    experiments_on_regulators[field+'_corr'] = experiments_on_regulators.apply(lambda x:get_corr_field_value(x,field),1)
    experiments_on_regulators['t']=1
    thr=0.05
    tmp = experiments_on_regulators.loc[experiments_on_regulators['qval']<thr]
    df_gr = tmp.groupby(['regulator']).agg({'qval':max,'abs_'+field:median,'t':sum,field+'_corr':max}).reset_index().rename(columns={field+'_corr':field+'_max','t':'num_of_sign_expers'})
    df_gr_1 = tmp.groupby(['regulator']).agg({field+'_corr':min}).reset_index().rename(columns={field+'_corr':field+'_min'})
    df_gr = pd.merge(df_gr,df_gr_1,how='left',on='regulator')
    df_gr['consistent_sign'] = (df_gr[field+'_min']*df_gr[field+'_max']>0).astype('int')
    tmp = experiments_on_regulators.groupby('regulator').agg({'t':sum}).reset_index().rename(columns={'t':'num_of_expers'})
    res = pd.merge(df_gr,tmp,how='left',on='regulator')
    tmp = experiments_on_regulators.loc[(~experiments_on_regulators['regulator'].isin(list(res['regulator'].unique())))]
    df_gr = tmp.groupby(['regulator']).agg({'qval':max,'abs_'+field:median,'t':sum,field+'_corr':max}).reset_index().rename(columns={field+'_corr':field+'_max','t':'num_of_expers'})
    df_gr_1 = tmp.groupby(['regulator']).agg({field+'_corr':min}).reset_index().rename(columns={field+'_corr':field+'_min'})
    df_gr = pd.merge(df_gr,df_gr_1,how='left',on='regulator')
    df_gr['consistent_sign'] = (df_gr[field+'_min']*df_gr[field+'_max']>0).astype('int')
    # consistent sign = 1 - если есть значимые, то только среди значимых; если нет - то среди всех доступных экспериментов
    df_gr['num_of_sign_expers']=0
    res = pd.concat([res[['regulator', 'num_of_expers', 'num_of_sign_expers', 'qval', 'abs_'+field, field+'_max', field+'_min', 'consistent_sign']],
                    df_gr[['regulator', 'num_of_expers', 'num_of_sign_expers', 'qval', 'abs_'+field, field+'_max', field+'_min', 'consistent_sign']]])
    res = res.sort_values(['consistent_sign','num_of_sign_expers','qval','abs_'+field,field+'_min',field+'_max'],ascending=[False,False,True,False,True,False]).reset_index(drop=True)
    res[['qval','abs_'+field, field+'_max', field+'_min']] = np.round(res[['qval','abs_'+field, field+'_max', field+'_min']],3)
    res.rename(columns={'abs_'+field:'median_abs_'+field},inplace=True)
    return res

# смотрим на исследуемые гены в данных по нокдаунам, нокаутам и оверэкспрессии RBP
def get_deseq_data_from_regulators_experiments(gene_name,SRA_directory,RBP_shRNA_KD_dir,rscript_path,regulators = None,exclude=None):
    # данные ENCODE по нокдаунам RBP
    shRNA_table = pd.read_csv('/home/magmir/TASS/RBP_shRNA_KD/shRNA_table.tsv',delimiter="\t",index_col=None,header=None)
    shRNA_table.columns = ['KD1','KD2','CTL1','CTL2','cell_line','target','tmp']
    shRNA_table.drop('tmp',1,inplace=True)
    # убираем эксперименты с неудавшимся нокдауном
    shRNA_table = shRNA_table.loc[~((shRNA_table['target']=='UPF1')&(shRNA_table['cell_line']=='HepG2'))]
    # дополнительные данные из sra
    public_data_on_AS_NMD = pd.read_csv('/home/magmir/TASS/NMD_regulation/public_data_on_AS_NMD.txt',delimiter="\t",index_col=None,header=0)    
    # по умолчанию мы не знаем, какие регуляторы у события, и нам надо проверить все возможные варианты. 
    if regulators is None or regulators==[]:
        public_data_on_AS_NMD_RBP_list = pd.read_csv('/home/magmir/TASS/NMD_regulation/public_data_on_AS_NMD_RBP_list.txt',delimiter="\t",index_col=None,header=None)
        regulators = list(np.unique(list(public_data_on_AS_NMD_RBP_list[0].unique())+list(shRNA_table['target'])))
        regulators.remove('UPF1') # на ингибирование NMD будем смотреть отдельно
    if exclude is not None:
        for regulator in exclude:
            if regulator in regulators:
                regulators.remove(regulator)    
    result = []
    i=1
    print('begin multiprocessing job.')
    with Manager() as manager:
        result = manager.list()
        p = Pool(multiprocessing.cpu_count()-5)
        for regulator in regulators:
            p.apply_async(get_deseq_data_for_regulator, args=(result,gene_name,regulator,shRNA_table,
                                                        RBP_shRNA_KD_dir,public_data_on_AS_NMD,SRA_directory))
        p.close()
        p.join()
        result = list(result)
    result = pd.DataFrame(result,columns=['regulator','Database','cell_line','experiment','pval','log2FC'])
    if len(result)>0:
        result['id'] = result.index
        result[['id','pval']].to_csv(
                '/home/magmir/TASS/NMD_regulation/temp/RBP_experiments_pvals.tsv', 
                              sep=str('\t'),index=None,header=True,encoding='utf-8')
        if os.path.isfile('/home/magmir/TASS/NMD_regulation/temp/RBP_experiments_qvals.tsv'):
            os.system('rm /home/magmir/TASS/NMD_regulation/temp/RBP_experiments_qvals.tsv')
        os.system(rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_q_values.r '+\
                  '/home/magmir/TASS/NMD_regulation/temp/RBP_experiments_pvals.tsv '+\
                 '/home/magmir/TASS/NMD_regulation/temp/RBP_experiments_qvals.tsv '+\
                 '/home/magmir/libs/R ')
        result_qvals = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/RBP_experiments_qvals.tsv',
                                            delimiter="\t",index_col=None,header=0)
        result = pd.merge(result,result_qvals.drop('pval',1),how='left',on='id')
        result.drop('id',1,inplace=True)
    else:
        result = pd.DataFrame(columns=['regulator','Database','cell_line','experiment','pval','log2FC','qval','lFDR'])
    return result    

def get_deseq_data_for_regulator(result,gene_name,regulator,shRNA_table,RBP_shRNA_KD_dir,public_data_on_AS_NMD,SRA_directory):
    # сначала ищем в данных ENCODE
    shRNA_df = shRNA_table.loc[shRNA_table['target']==regulator]
    if len(shRNA_df)>0:
        for index, row in shRNA_df.iterrows():
            if os.path.isfile(RBP_shRNA_KD_dir+regulator+'/'+row['cell_line']+'/deseq2/res.tsv.gz'):
                
                df_reg = pd.read_csv(RBP_shRNA_KD_dir+regulator+'/'+row['cell_line']+'/deseq2/res.tsv.gz',
                                        delimiter="\t",index_col=None,header=0,compression='gzip')
                # проверяем, что есть эффект нокдауна. Если ошибочно эффект в другую сторону - меняем знак
                x = df_reg.loc[df_reg['gene_name']==regulator].iloc[0]
                if (x['log2FoldChange']>0.5):
                    coef = -1
                elif (abs(x['log2FoldChange'])<=0.5):
                    result.append([regulator,'ENCODE',row['cell_line'],regulator+' shRNA KD (not efficient)',1,0])
                    continue
                    # неудачный нокдаун
                else:
                    coef = 1
                df_reg = df_reg.loc[df_reg['gene_name']==gene_name]
                if len(df_reg)>0:
                    # нам нужен не FDR, а p-value от deseq, которое мы потом скорректируем сами
                    PValue = df_reg.iloc[0]['pvalue']
                    log2FC = df_reg.iloc[0]['log2FoldChange']*coef
                    result.append([regulator,'ENCODE',row['cell_line'],regulator+' shRNA KD',PValue,log2FC])
                else:
                    result.append([regulator,'ENCODE',row['cell_line'],regulator+' shRNA KD',1,0])
    # дополнительно смотрим в sra
    public_data_on_AS_NMD_df = public_data_on_AS_NMD.loc[public_data_on_AS_NMD['Experiments'].str.contains(' '+regulator)]
    if len(public_data_on_AS_NMD_df)>0:
        for index, row in public_data_on_AS_NMD_df.iterrows():
            if os.path.isfile(SRA_directory+row['SRP']+'/deseq2/hg19/'+('' if pd.isna(row['sub_path']) else row['sub_path'])+'res.tsv.gz'):
                df_reg = pd.read_csv(SRA_directory+row['SRP']+'/deseq2/hg19/'+('' if pd.isna(row['sub_path']) else row['sub_path'])+'res.tsv.gz',
                                   delimiter="\t",index_col=None,header=0,compression='gzip')
                # проверяем, что эффект от нокдауна или оверэкспрессии в нужную сторону
                if regulator=='CHX' or row['Experiments']==' UPF1 siRNA + XRN1 siRNA' or ('deletion mutant' in row['Experiments']) or ('KO' in row['Experiments']):
                    coef=1
                else:
                    x = df_reg.loc[df_reg['gene_name']==regulator].iloc[0]
                    if (regulator+' OE' in row['Experiments']):
                        expected_log2FC = 1
                    elif (regulator+' KD' in row['Experiments']) or (regulator+' siRNA' in row['Experiments']) or (regulator+' shRNA' in row['Experiments']):
                        expected_log2FC = -1
                    else:
                        expected_log2FC = 0
                    if (x['log2FoldChange']*expected_log2FC<-0.5):
                        coef = -1
                    elif (abs(x['log2FoldChange'])<=0.5):
                        # неудавшийся эксперимент
                        result.append([regulator,row['SRP'],row['Cell lines'],row['Experiments']+' (not efficient)',1,0])
                        continue
                    else:
                        coef = 1
                df_reg = df_reg.loc[df_reg['gene_name']==gene_name]
                if len(df_reg)>0:
                    PValue = df_reg.iloc[0]['pvalue']
                    log2FC = df_reg.iloc[0]['log2FoldChange']*coef
                    result.append([regulator,row['SRP'],row['Cell lines'],row['Experiments'],PValue,log2FC])
                else:
                    result.append([regulator,row['SRP'],row['Cell lines'],row['Experiments'],1,0])

def get_proteomicsDB_expr_data(gene):
    all_ProtDB_tissues_x_smtsd = pd.read_csv(open('/home/magmir/TASS/ProteomicsDB/all_ProtDB_tissues_x_smtsd.tsv'),delimiter="\t",
                                   index_col=None,header=0)
    all_ProtDB_tissues_x_smtsd = all_ProtDB_tissues_x_smtsd.loc[all_ProtDB_tissues_x_smtsd['smtsd']!='-']
    if not os.path.isfile('/home/magmir/TASS/ProteomicsDB/'+gene+'.json'):
        Uniprot_to_gene_name = pd.read_csv(open('/home/magmir/TASS/ProteomicsDB/Uniprot_to_gene_name.tsv'),delimiter="\t",
                                   index_col=None,header=None)
        Uniprot_to_gene_name.columns=['PROTEINFILTER','tmp','gene_name']
        tmp1 = Uniprot_to_gene_name.loc[Uniprot_to_gene_name['gene_name']==gene]
        if len(tmp1)==0:
            return None
        PROTEINFILTER = tmp1.iloc[0]['PROTEINFILTER']
        command = """curl -H "Accept: application/json" -o /home/magmir/TASS/ProteomicsDB/"""+gene+""".json """+\
        """"https://www.proteomicsdb.org/proteomicsdb/logic/api/proteinexpression.xsodata/InputParams("""+\
                  """PROTEINFILTER='"""+PROTEINFILTER+"""',"""+\
                  """MS_LEVEL=1,"""+\
                  """TISSUE_ID_SELECTION='',"""+\
                  """TISSUE_CATEGORY_SELECTION='tissue;fluid',"""+\
                  """SCOPE_SELECTION=1,"""+\
                  """GROUP_BY_TISSUE=0,CALCULATION_METHOD=0,EXP_ID=-1)/Results?$select=TISSUE_NAME,PROJECT_NAME,NORMALIZED_INTENSITY&$format=json"
                  """
        os.system(command)
    if os.path.isfile('/home/magmir/TASS/ProteomicsDB/'+gene+'.json'):
        with open('/home/magmir/TASS/ProteomicsDB/'+gene+'.json') as fh:
            protdf = pd.DataFrame(json.load(fh)['d']['results'])
            if len(protdf)>0:
                protdf = protdf[['TISSUE_NAME','PROJECT_NAME','NORMALIZED_INTENSITY']]
                protdf['NORMALIZED_INTENSITY'] = protdf['NORMALIZED_INTENSITY'].astype('float')
                protdf = protdf.loc[protdf['TISSUE_NAME'].isin(list(all_ProtDB_tissues_x_smtsd['TISSUE_NAME'].unique()))]
                protdf['NORMALIZED_INTENSITY'] = protdf['NORMALIZED_INTENSITY'].fillna(0)
                return protdf
            else:
                return None
    else:
        print('no proteomics data for '+gene)
        return None
# figure psi vs gene expression in upper and lower clusters    
def fig_psi_vs_gene_expr_in_clusters(junction_count_data,gtex_psi_vs_expr,step=2):
    a = junction_count_data.loc[junction_count_data['cluster']=='LQ']
    b = junction_count_data.loc[junction_count_data['cluster']=='UQ']

    sns.set(font_scale=1)
    sns.set_style("whitegrid",{'axes.grid' : False})
    sns.despine()
    fig_psi_vs_gene_clusters, axes = plt.subplots(1, 3, sharey=False, figsize=(7,0.8),gridspec_kw={'width_ratios': [4,2,2]})
    ax = sns.histplot(ax=axes[0],data=junction_count_data,x='psi',hue='cluster',
                      hue_order = ['LQ','MED','UQ'],palette=['blue','grey','orange'],
                      kde=False,stat='density',legend=False,bins=get_bins(a,b,step))
    ax.set_title(label=gtex_psi_vs_expr['gene_name'],loc='left')
#     x1,x2,y,h = a['psi'].median(),b['psi'].median(),-0.5,-0.2
#     ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5,color='black')
#     ax.text(0.5*(x1+x2), y+h, str(np.round(b['psi'].median()-a['psi'].median(),2)), 
#             ha='center', va='bottom', color='black')    
    x_min = -0.05
    x_max = 1.05
    if a['psi'].median()>0.8:
        x_min=0.5
    elif b['psi'].median()<0.2:
        x_max=0.5
    ax.set(xlabel=r'$\psi$',xlim=(x_min,x_max),ylabel='',yticklabels=[])
    
    ax = sns.boxplot(ax=axes[1],data=junction_count_data,x='global_expr',y='cluster',
                     order=['LQ','UQ'],palette=['blue','orange'],saturation=1,
                      showfliers=False)
    x1,x2,y,h = min(a['global_expr'].median(),b['global_expr'].median()),max(a['global_expr'].median(),b['global_expr'].median()),-0.5,-0.2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5,color='black')
    log2FC = np.round(gtex_psi_vs_expr['log2FC_global'],1)
    zscore = gtex_psi_vs_expr['zscore_global']
    pval = 2*(1-stats.norm.cdf(abs(zscore)))
    star = get_pvalue_star(pval)
    if (log2FC<0) and (pval<0.05):
        color = 'green'
    elif (log2FC>0) and (pval<0.05):
        color = 'red'
    else:
        color='black'
    ax.text(0.5*(x1+x2), y+h, str(log2FC)+star, 
            ha='center', va='bottom', color=color)
    ax.set(yticklabels=[],ylabel='',xlabel='global expr.')

    ax = sns.boxplot(ax=axes[2],data=junction_count_data,x='local_expr',y='cluster',
                     order=['LQ','UQ'],palette=['blue','orange'],saturation=1,
                      showfliers=False)
    x1,x2,y,h = min(a['local_expr'].median(),b['local_expr'].median()),max(a['local_expr'].median(),b['local_expr'].median()),-0.5,-0.2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5,color='black')
    log2FC = np.round(gtex_psi_vs_expr['log2FC_local'],1)
    zscore = gtex_psi_vs_expr['zscore_local']
    pval = 2*(1-stats.norm.cdf(abs(zscore)))
    star = get_pvalue_star(pval)
    if (log2FC<0) and (pval<0.05):
        color = 'green'
    elif (log2FC>0) and (pval<0.05):
        color = 'red'
    else:
        color='black'
    ax.text(0.5*(x1+x2), y+h, str(log2FC)+star, 
            ha='center', va='bottom', color=color)
    ax.set(yticklabels=[],ylabel='',xlabel='local expr.')
    
    return fig_psi_vs_gene_clusters
    
    
# figure psi vs regulators expression in upper and lower clusters    
def fig_psi_vs_reg_expr_in_clusters(junction_count_data,regulators_gtex=None,regulators_to_show=None,expected_direction=None):
    # информация о всех регуляторах в списке regulators_to_show должна содержаться в regulators_gtex
    if regulators_to_show is not None:
        if len(regulators_to_show)!=len(expected_direction):
            return None
    a = junction_count_data.loc[junction_count_data['cluster']=='LQ']
    b = junction_count_data.loc[junction_count_data['cluster']=='UQ']
    if regulators_gtex is not None:
        full_list = list(regulators_gtex['regulator'].unique())
        if regulators_to_show==None:
            regulators_to_show=full_list
            expected_direction=[0]*len(full_list)
        else:
            for regulator in regulators_to_show:
                if regulator not in full_list:
                    print('regulator '+regulator+' is not in the list')
                    return None
    else:
        regulators_to_show=[]
        expected_direction=[]
        
    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    fig_psi_vs_reg_clusters, axes = plt.subplots(1, 1+len(regulators_to_show), sharey=False, figsize=(2+len(regulators_to_show)*2,1))
    ax = sns.histplot(ax=(axes[0] if len(regulators_to_show)>0 else None),data=junction_count_data,x='psi',hue='cluster',
                      hue_order = ['LQ','MED','UQ'],palette=['blue','grey','orange'],
                      kde=False,stat='density',legend=False)
    x1,x2,y,h = a['psi'].median(),b['psi'].median(),-0.5,-0.2
    ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5,color='black')
    ax.text(0.5*(x1+x2), y+h, str(np.round(b['psi'].median()-a['psi'].median(),2)), 
            ha='center', va='bottom', color='black')    
    ax.set(xlabel=r'$\psi$',xlim=(-0.0001,1.0001))
    
    L = 0
    U = 0
    for regulator in regulators_to_show:
        L = min(junction_count_data[regulator].quantile(0.25)-1.6*(junction_count_data[regulator].quantile(0.75)-junction_count_data[regulator].quantile(0.25)),L)
        U = max(junction_count_data[regulator].quantile(0.75)+1.6*(junction_count_data[regulator].quantile(0.75)-junction_count_data[regulator].quantile(0.25)),U)
    
    i=1
    for regulator in regulators_to_show:
        ax = sns.boxplot(ax=axes[i],data=junction_count_data,x=regulator,y='cluster',
                         order=['LQ','UQ'],palette=['blue','orange'],
                          showfliers=False)
        x1,x2,y,h = min(a[regulator].median(),b[regulator].median()),max(a[regulator].median(),b[regulator].median()),-0.5,-0.2
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5,color='black')
        cur = regulators_gtex.loc[regulators_gtex['regulator']==regulator].iloc[0]
        log2FC = cur['log2FC']
        pval = cur['pval_clust_adj']
        star = get_pvalue_star(pval)
        if log2FC*expected_direction[i-1]>0:
            color = 'green'
        elif log2FC*expected_direction[i-1]<0:
            color = 'red'
        else:
            color='black'
        ax.text(0.5*(x1+x2), y+h, str(log2FC)+star, 
                ha='center', va='bottom', color=color)
        ax.set(yticklabels=[],ylabel='',xlabel=regulator,xlim=(L,U))
        i=i+1
    return fig_psi_vs_reg_clusters

# figure psi vs regulators expression in upper and lower clusters in regulators    
def fig_psi_vs_reg_expr_in_clusters_backwards(junction_count_data,regulators_gtex,regulators_to_show=None,expected_direction=None):
    # информация о всех регуляторах в списке regulators_to_show должна содержаться в regulators_gtex
    if regulators_to_show is not None:
        if len(regulators_to_show)!=len(expected_direction):
            return None
    full_list = list(regulators_gtex['regulator'].unique())
    if regulators_to_show==None:
        regulators_to_show=full_list
        expected_direction=[0]*len(full_list)
    else:
        for regulator in regulators_to_show:
            if regulator not in full_list:
                print('regulator '+regulator+' is not in the list')
                return None
    L = 0
    U = 0
    for regulator in regulators_to_show:
        L = min(junction_count_data[regulator].quantile(0.25)-1.6*(junction_count_data[regulator].quantile(0.75)-junction_count_data[regulator].quantile(0.25)),L)
        U = max(junction_count_data[regulator].quantile(0.75)+1.6*(junction_count_data[regulator].quantile(0.75)-junction_count_data[regulator].quantile(0.25)),U)
    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    fig_psi_vs_reg_clusters, axes = plt.subplots(len(regulators_to_show), 2, sharey=False,sharex=False, figsize=(5,len(regulators_to_show)*1.5))
    i=0
    for regulator in regulators_to_show:
        a = junction_count_data.loc[junction_count_data['cluster_'+regulator]=='LQ']
        b = junction_count_data.loc[junction_count_data['cluster_'+regulator]=='UQ']
        
        ax = sns.histplot(ax=axes[i][0],data=junction_count_data,x=regulator,hue='cluster_'+regulator,
                      hue_order = ['LQ','MED','UQ'],palette=['blue','grey','orange'],
                      kde=False,stat='density',legend=False)
        ax.set(ylabel=regulator,xlabel='',title='',xlim=(L,U))
        if i==0:
            ax.set(title='expression')
        ax = sns.boxplot(ax=axes[i][1],data=junction_count_data,x='psi',y='cluster_'+regulator,
                         order=['LQ','UQ'],palette=['blue','orange'],
                          showfliers=False)
        x1,x2,y,h = min(a['psi'].median(),b['psi'].median()),max(a['psi'].median(),b['psi'].median()),-0.5,-0.2
        ax.plot([x1, x1, x2, x2], [y, y+h, y+h, y], lw=0.5,color='black')
        cur = regulators_gtex.loc[regulators_gtex['regulator']==regulator].iloc[0]
        delta_psi = cur['back_delta_psi']
        pval = cur['back_pval_clust_adj']
        star = get_pvalue_star(pval)
        if delta_psi*expected_direction[i]>0:
            color = 'green'
        elif delta_psi*expected_direction[i]<0:
            color = 'red'
        else:
            color='black'        
        ax.text(0.5*(x1+x2), y+h, str(delta_psi)+star, 
                ha='center', va='bottom', color=color)
        ax.set(yticklabels=[],ylabel='',xlabel='',xlim=(-0.0001,1.0001))
        if i==0:
            ax.set(title=r'$\psi$')
        i=i+1
    return fig_psi_vs_reg_clusters

def fig_psi_vs_ts_regulator_expression_scatter(df,regulators):
    GTEX_color_codes = pd.read_csv('/home/magmir/TASS/GTEX/GTEX_color_codes.tsv',delimiter="\t",index_col=None,header=0)    
    GTEX_color_codes.rename(columns={'SMTSD':'smtsd'},inplace=True)
    df = pd.merge(df,GTEX_color_codes[['smtsd','color_code','color_code_alt']],how='left',on='smtsd')

    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    fig_TS_scatter, axes = plt.subplots(len(regulators), 1, sharey=True, figsize=(5,5*len(regulators)))
    i=0
    for regulator in regulators:
        df.sort_values('log2FC_'+regulator,ascending=False,inplace=True)
        df1 = df.loc[(df['pval_'+regulator]<0.05)]
        df2 = df.loc[~(df['pval_'+regulator]<0.05)]
        order = list(df1['smtsd'])
        palette = list(df1['color_code_alt'])
        ax = sns.scatterplot(ax=(axes[i] if len(regulators)>1 else None),data = df1,x='log2FC_'+regulator,y='delta_psi',hue='smtsd',hue_order=order,palette=palette)
        ax = sns.scatterplot(ax=(axes[i] if len(regulators)>1 else None),data = df2,x='log2FC_'+regulator,y='delta_psi',legend=False,color='grey')
        ax.set(ylabel=r'$\Delta\psi$',xlabel=r'$log_2FC$ '+regulator)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, title="", borderaxespad=0.)
        i=i+1
    return fig_TS_scatter

def fig_psi_vs_ts_regulator_expression_boxplot(junction_count_data,regulators=[],tissue_spec_res=None,expected_direction=[]):
    if len(regulators)!=len(expected_direction):
        return None
    GTEX_color_codes = pd.read_csv('/home/magmir/TASS/GTEX/GTEX_color_codes.tsv',delimiter="\t",index_col=None,header=0)    
    GTEX_color_codes.rename(columns={'SMTSD':'smtsd'},inplace=True)
    GTEX_color_codes = GTEX_color_codes[['smtsd','color_code_alt']]
    a = pd.DataFrame(['other tissues','white']).transpose()
    a.columns = ['smtsd','color_code_alt']
    GTEX_color_codes = pd.concat([GTEX_color_codes,a])
    if tissue_spec_res is not None:
        sign_tissues = list(tissue_spec_res.loc[tissue_spec_res['delta_psi_qval']<0.05]['smtsd'])
        junction_count_data['tissue'] = junction_count_data.apply(lambda x:get_tissue(x,sign_tissues),1)
    else:
        junction_count_data['tissue'] = junction_count_data['smtsd']
    
    gr = junction_count_data.groupby(['tissue']).agg({'psi':median}).reset_index().rename(columns={'tissue':'smtsd'})
    gr = pd.merge(gr,GTEX_color_codes,how='left',on='smtsd')
    gr.sort_values('psi',inplace=True)
    
    order = list(gr['smtsd'])
    palette = list(gr['color_code_alt'])
    
    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    fig_TS_boxplot, axes = plt.subplots(1,1+len(regulators),sharey=True, figsize=(3+3*len(regulators),len(order)*0.6))
    ax = sns.boxplot(ax=(axes[0] if len(regulators)>0 else None),data=junction_count_data,x='psi',y='tissue',
                     order=order,palette=palette,showfliers=False)
    if tissue_spec_res is not None:
        j=0
        for tissue in order:
            if tissue!='other tissues':
                delta_psi_qval = tissue_spec_res.loc[tissue_spec_res['smtsd']==tissue]['delta_psi_qval'].iloc[0]
                star = get_pvalue_star(delta_psi_qval)
                delta_psi = tissue_spec_res.loc[tissue_spec_res['smtsd']==tissue]['delta_psi'].iloc[0]
                if delta_psi>0:
                    sign = '+'
                else:
                    sign = '-'
                l = junction_count_data.loc[junction_count_data['smtsd']==tissue]['psi']
                whisker_lim = 1.5*(l.quantile(0.75)-l.quantile(0.25))
                x = np.max(l[l <= (l.quantile(0.75) + whisker_lim)])
                y=j
                ax.text(x,y,' '+star+'('+sign+')',ha='left', va='center')
            j=j+1
    ax.set(xlabel=r'$\psi$',ylabel='')
    i=1
    for regulator in regulators:
        ax = sns.boxplot(ax=axes[i],data = junction_count_data,x=regulator,y='tissue',order=order,palette=palette,showfliers=False)
        j=0
        for tissue in order:
            if tissue!='other tissues':
                delta_psi_qval = tissue_spec_res.loc[tissue_spec_res['smtsd']==tissue]['delta_psi_qval'].iloc[0]
                delta_psi = tissue_spec_res.loc[tissue_spec_res['smtsd']==tissue]['delta_psi'].iloc[0]
                logFC_qval = tissue_spec_res.loc[tissue_spec_res['smtsd']==tissue]['qval_'+regulator].iloc[0]
                logFC = tissue_spec_res.loc[tissue_spec_res['smtsd']==tissue]['log2FC_'+regulator].iloc[0]
                star = get_pvalue_star(logFC_qval)
                font_color = 'grey'
                sign = ''
                if (logFC_qval<0.05) and (delta_psi_qval<0.05):
                    if expected_direction[i-1]*logFC*delta_psi>0:
                        font_color='green'
                    else:
                        font_color='red'
                    if logFC>0:
                        sign='(+)'
                    else:
                        sign='(-)'
                l = junction_count_data.loc[junction_count_data['smtsd']==tissue][regulator].astype('float')
                whisker_lim = 1.5*(l.quantile(0.75)-l.quantile(0.25))
                x = np.max(l[l <= (l.quantile(0.75) + whisker_lim)])
                y=j
                ax.text(x,y,' '+star+sign,ha='left', va='center',color=font_color)
            j=j+1
        ax.set(ylabel='',xlabel=regulator)
        i=i+1
    return fig_TS_boxplot

def fig_psi_vs_gene_expr_and_proteomics_in_tissues(junction_count_data,gene_name,protdf,rscript_path,exp_sign):
    all_ProtDB_tissues_x_smtsd = pd.read_csv(open('/home/magmir/TASS/ProteomicsDB/all_ProtDB_tissues_x_smtsd.tsv'),delimiter="\t",
                                   index_col=None,header=0)
    all_ProtDB_tissues_x_smtsd = all_ProtDB_tissues_x_smtsd.loc[all_ProtDB_tissues_x_smtsd['smtsd']!='-']
    all_ProtDB_tissues_x_smtsd = all_ProtDB_tissues_x_smtsd.loc[all_ProtDB_tissues_x_smtsd['TISSUE_NAME'].isin(list(protdf['TISSUE_NAME'].unique()))]
    
    junction_count_data = pd.merge(junction_count_data,all_ProtDB_tissues_x_smtsd,how='inner',on='smtsd')
    
    res = tissue_specificity_of_psi_and_regulator_expression(junction_count_data,[gene_name,'local_expr'],rscript_path,'TISSUE_NAME')
    junction_count_data = res[0]
    tissue_spec_res = res[1]
    
    gr = junction_count_data.groupby(['TISSUE_NAME','color_code']).agg({'psi':median}).reset_index()
    gr = gr.sort_values('psi',ascending=True).reset_index(drop=True)
    gr['order_numb'] = gr.index
    order = gr['TISSUE_NAME']
    palette = gr['color_code']
    
    junction_count_data = pd.merge(junction_count_data,gr[['TISSUE_NAME','order_numb']],how='right',on='TISSUE_NAME')
    junction_count_data[['order_numb','psi',gene_name,'local_expr']].to_csv(
            '/home/magmir/TASS/NMD_regulation/temp/psi_and_gene_mRNA_expression.tsv', 
                          sep=str('\t'),index=None,header=True,encoding='utf-8')
    stats_res = []
    for field in ['psi',gene_name,'local_expr']:
        if os.path.isfile('/home/magmir/TASS/NMD_regulation/temp/jonckheere_pval_'+str(field)+'.txt'):
            os.system('rm /home/magmir/TASS/NMD_regulation/temp/jonckheere_pval_'+str(field)+'.txt') 
        if field=='psi':
            alt='increasing'
        else:
            alt='decreasing'
        command = rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_jonckheere_test.r '+\
        '/home/magmir/TASS/NMD_regulation/temp/psi_and_gene_mRNA_expression.tsv '+\
        '/home/magmir/TASS/NMD_regulation/temp/jonckheere_pval_'+str(field)+'.txt '+\
        field+' '+\
        'order_numb '+\
        alt+' '+\
        '/home/magmir/libs/R'
        os.system(command)
        jonckheere_pval = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/jonckheere_pval_'+str(field)+'.txt',
                                      delimiter="\t",index_col=None,header=None)
        jonckheere_pval = jonckheere_pval.iloc[0][0]
        stats_res.append(['jonckheere_pval_'+field,jonckheere_pval])
    
    stats_res = pd.DataFrame(stats_res,columns=['stat','value'])
    
    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    fig_TS_boxplot, axes = plt.subplots(1,4,sharey=True, figsize=(16,len(order)*0.3))
    ax = sns.boxplot(ax=axes[0],data=junction_count_data,x='psi',y='TISSUE_NAME',order=order,palette=palette,showfliers=False)
    j=0
    for tissue in order:
        delta_psi_qval = tissue_spec_res.loc[tissue_spec_res['TISSUE_NAME']==tissue]['delta_psi_qval'].iloc[0]
        star = get_pvalue_star(delta_psi_qval)
        delta_psi = tissue_spec_res.loc[tissue_spec_res['TISSUE_NAME']==tissue]['delta_psi'].iloc[0]
        if delta_psi>0:
            sign = '+'
        else:
            sign = '-'
        l = junction_count_data.loc[junction_count_data['TISSUE_NAME']==tissue]['psi']
        whisker_lim = 1.5*(l.quantile(0.75)-l.quantile(0.25))
        x = np.max(l[l <= (l.quantile(0.75) + whisker_lim)])
        y=j
        ax.text(x,y,' '+star+'('+sign+')',ha='left', va='center')
        j=j+1
    trend_pval = np.round(stats_res.loc[stats_res['stat']=='jonckheere_pval_psi'].iloc[0]['value'],2)
    ax.set(xlabel=r'$\psi$',ylabel='tissue',title='trend test pval:'+str(trend_pval))
    
    ax = sns.boxplot(ax=axes[1],data = junction_count_data,x=gene_name,y='TISSUE_NAME',order=order,palette=palette,showfliers=False)
    j=0
    for tissue in order:
        delta_psi_qval = tissue_spec_res.loc[tissue_spec_res['TISSUE_NAME']==tissue]['delta_psi_qval'].iloc[0]
        delta_psi = tissue_spec_res.loc[tissue_spec_res['TISSUE_NAME']==tissue]['delta_psi'].iloc[0]
        logFC_qval = tissue_spec_res.loc[tissue_spec_res['TISSUE_NAME']==tissue]['qval_'+gene_name].iloc[0]
        logFC = tissue_spec_res.loc[tissue_spec_res['TISSUE_NAME']==tissue]['log2FC_'+gene_name].iloc[0]
        star = get_pvalue_star(logFC_qval)
        font_color = 'grey'
        sign = ''
        if (logFC_qval<0.05) and (delta_psi_qval<0.05):
            if exp_sign*logFC*delta_psi>0:
                font_color='green'
            else:
                font_color='red'
            if logFC>0:
                sign='(+)'
            else:
                sign='(-)'
        l = junction_count_data.loc[junction_count_data['TISSUE_NAME']==tissue][gene_name].astype('float')
        whisker_lim = 1.5*(l.quantile(0.75)-l.quantile(0.25))
        x = np.max(l[l <= (l.quantile(0.75) + whisker_lim)])
        y=j
        ax.text(x,y,' '+star+sign,ha='left', va='center',color=font_color)
        j=j+1
    trend_pval = np.round(stats_res.loc[stats_res['stat']=='jonckheere_pval_'+gene_name].iloc[0]['value'],2)
    ax.set(ylabel='',xlabel='global mRNA expr.',title='trend test pval:'+str(trend_pval))
    
    ax = sns.boxplot(ax=axes[2],data = junction_count_data,x='local_expr',y='TISSUE_NAME',order=order,palette=palette,showfliers=False)
    j=0
    for tissue in order:
        delta_psi_qval = tissue_spec_res.loc[tissue_spec_res['TISSUE_NAME']==tissue]['delta_psi_qval'].iloc[0]
        delta_psi = tissue_spec_res.loc[tissue_spec_res['TISSUE_NAME']==tissue]['delta_psi'].iloc[0]
        logFC_qval = tissue_spec_res.loc[tissue_spec_res['TISSUE_NAME']==tissue]['qval_local_expr'].iloc[0]
        logFC = tissue_spec_res.loc[tissue_spec_res['TISSUE_NAME']==tissue]['log2FC_local_expr'].iloc[0]
        star = get_pvalue_star(logFC_qval)
        font_color = 'grey'
        sign = ''
        if (logFC_qval<0.05) and (delta_psi_qval<0.05):
            if exp_sign*logFC*delta_psi>0:
                font_color='green'
            else:
                font_color='red'
            if logFC>0:
                sign='(+)'
            else:
                sign='(-)'
        l = junction_count_data.loc[junction_count_data['TISSUE_NAME']==tissue]['local_expr'].astype('float')
        whisker_lim = 1.5*(l.quantile(0.75)-l.quantile(0.25))
        x = np.max(l[l <= (l.quantile(0.75) + whisker_lim)])
        y=j
        ax.text(x,y,' '+star+sign,ha='left', va='center',color=font_color)
        j=j+1
    trend_pval = np.round(stats_res.loc[stats_res['stat']=='jonckheere_pval_local_expr'].iloc[0]['value'],2)
    ax.set(ylabel='',xlabel='local mRNA expr.',title='trend test pval:'+str(trend_pval))
    
    ax = sns.swarmplot(ax=axes[3],data = protdf,x='NORMALIZED_INTENSITY',y='TISSUE_NAME',order=order,palette=palette)
    protdf = pd.merge(protdf,gr[['TISSUE_NAME','order_numb']],how='right',on='TISSUE_NAME')
    protdf.to_csv(
            '/home/magmir/TASS/NMD_regulation/temp/prot_expression.tsv', 
                          sep=str('\t'),index=None,header=True,encoding='utf-8')
    if os.path.isfile('/home/magmir/TASS/NMD_regulation/temp/jonckheere_pval.txt'):
        os.system('rm /home/magmir/TASS/NMD_regulation/temp/jonckheere_pval.txt') 
    command = rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_jonckheere_test.r '+\
    '/home/magmir/TASS/NMD_regulation/temp/prot_expression.tsv '+\
    '/home/magmir/TASS/NMD_regulation/temp/jonckheere_pval.txt '+\
    'NORMALIZED_INTENSITY '+\
    'order_numb '+\
    'decreasing '+\
    '/home/magmir/libs/R'
    os.system(command)
    jonckheere_pval = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/jonckheere_pval.txt',delimiter="\t",index_col=None,header=None)
    jonckheere_pval = jonckheere_pval.iloc[0][0]
    ax.set(ylabel='',xlabel='protein expr.',title='trend test pval:'+str(np.round(jonckheere_pval,2)))
    stats_df = pd.DataFrame([['jonckheere_pval_protein'],[jonckheere_pval]]).transpose()
    stats_df.columns = ['stat','value']
    stats_res = pd.concat([stats_res,stats_df])
    return [fig_TS_boxplot,stats_res]

def fig_psi_vs_gene_expr_in_tissues(junction_count_data,gene_name):
    
    GTEX_color_codes = pd.read_csv('/home/magmir/TASS/GTEX/GTEX_color_codes.tsv',delimiter="\t",index_col=None,header=0)    
    GTEX_color_codes.rename(columns={'SMTSD':'tissue'},inplace=True)
    GTEX_color_codes = GTEX_color_codes[['tissue','color_code']]
    a = pd.DataFrame(['other tissues','white']).transpose()
    a.columns = ['tissue','color_code']
    GTEX_color_codes = pd.concat([GTEX_color_codes,a])
    
    # если слишком много тканей, то показываем топ 7 и боттом 7
    if len(junction_count_data['smtsd'].unique())>15:
        gr = junction_count_data.groupby(['smtsd']).agg({'psi':median}).reset_index().sort_values('psi',ascending=True)
        smtsd_to_show = list(gr.head(7)['smtsd'])+list(gr.tail(7)['smtsd'])
        junction_count_data['tissue'] = junction_count_data.apply(lambda x:get_tissue(x,smtsd_to_show),1)        
    else:
        junction_count_data['tissue'] = junction_count_data['smtsd']
    
    gr = junction_count_data.groupby(['tissue']).agg({'psi':median}).reset_index()
    gr = pd.merge(gr,GTEX_color_codes,how='left',on='tissue')
    gr.sort_values('psi',inplace=True)
    order = list(gr['tissue'])
    palette = list(gr['color_code'])
        
    sns.set(font_scale=1)
    sns.set_style("whitegrid")
    fig_TS_boxplot, axes = plt.subplots(1,3,sharey=True, figsize=(6,len(order)*0.3))
    ax = sns.boxplot(ax=axes[0],data=junction_count_data,x='psi',y='tissue',order=order,palette=palette,showfliers=False)
    ax.set(xlabel=r'$\psi$',ylabel='tissue')
    
    ax = sns.boxplot(ax=axes[1],data = junction_count_data,x=gene_name,y='tissue',order=order,palette=palette,showfliers=False)
    ax.set(ylabel='',xlabel='global mRNA expr.')
    
    ax = sns.boxplot(ax=axes[2],data = junction_count_data,x='local_expr',y='tissue',order=order,palette=palette,showfliers=False)
    ax.set(ylabel='',xlabel='local mRNA expr.')    
    return fig_TS_boxplot


def run_ASNMD_analysis(gene_name,input_df,AS_type,known_regulators,directory,find_novel_regulators=False,block_NMD_exp=False,psi_range_thr=0):
    os.system('mkdir -p '+directory)
    # проверяем, что репрессия NMD приводит к апрегуляции NMD изоформы 
    try:
        experiments_on_NMD_repression = get_rMATS_data_from_NMD_repression(input_df,AS_type,SRA_directory='/home/magmir/TASS/DATA/SRA/',
                                                RBP_shRNA_KD_dir='/home/magmir/TASS/DATA/RBP_shRNA_KD_hg19/',
                                                                          rscript_path = '/mnt/lustre/tools/R/R-3.6.3/bin/')
        experiments_on_NMD_repression = experiments_on_NMD_repression[['regulator','Database','cell_line','experiment','delta_psi','qval','lFDR']]
        experiments_on_NMD_repression[['delta_psi','qval','lFDR']] = np.round(experiments_on_NMD_repression[['delta_psi','qval','lFDR']],3)
        experiments_on_NMD_repression = experiments_on_NMD_repression.sort_values(['qval','delta_psi'],ascending=[True,False])
        experiments_on_NMD_repression.to_csv(
            directory+'/experiments_on_NMD_repression.tsv', 
                              sep=str('\t'),index=None,header=True,encoding='utf-8')
        print(directory+' done NMD repression experiments (0)')
        # если не хотим дальше считать, раз не найден эффект апрегуляции NMD изоформы в репрессии NMD
        if block_NMD_exp==True:
            if len(experiments_on_NMD_repression.loc[(experiments_on_NMD_repression['qval']<0.05)&(experiments_on_NMD_repression['delta_psi']>0)])==0:
                print(directory+' no psi upregulation in NMD repression')
                return None
    except Exception as e:
        print(e)
        print(directory+' failed NMD repression experiments (0)')
    # проверяем, есть ли отрицательная ассоциация между включением NMD изоформы и экспрессией гена по всем образцам GTEx
    try:
        junction_count_data = get_psi_in_gtex(input_df,['Testis'])

        res = compare_psi_with_gene_expression_gtex(junction_count_data,gene_name,AS_type)
        junction_count_data = res[0]
        gtex_psi_vs_expr = res[1]
        if len(junction_count_data)>0 and len(gtex_psi_vs_expr)>0:
            fig = fig_psi_vs_gene_expr_in_clusters(junction_count_data,gene_name,gtex_psi_vs_expr)
            fig.savefig(directory+'/fig_psi_vs_gene_expr_in_clusters.png',bbox_inches="tight",dpi=300)
            psi_L = gtex_psi_vs_expr['psi_L'].iloc[0]
            psi_U = gtex_psi_vs_expr['psi_U'].iloc[0]
            psi_range = psi_U-psi_L
        else:
            psi_range=-1
            # нет смысла считать дальше, т.к. экспрессия в образцах GTEx очень мала            
        # если не хотим дальше считать, раз интервал пси слишком мал
        if psi_range<psi_range_thr:
            print(directory+' psi range is too narrow')
            return None
    except Exception as e:
        print(e)
        print(directory+' failed psi vs mRNA expression in gtex samples (1)')
    # проверяем, есть ли ткане-специфическая отрицательная ассоциация между включением NMD изоформы и экспрессией гена, в т.ч. в белке
    try:
        protdf = get_proteomicsDB_expr_data(gene_name)
        res = fig_psi_vs_gene_expr_and_proteomics_in_tissues(junction_count_data,gene_name,protdf,rscript_path='/mnt/lustre/tools/R/R-3.6.3/bin/',exp_sign=-1)
        fig = res[0]
        stats_df = res[1]
        fig.savefig(directory+'/fig_psi_vs_gene_expr_and_proteomics_in_tissues.png',bbox_inches="tight",dpi=300)
        
        tmp = stats_df.transpose()
        tmp.columns = list(tmp.loc['stat'])
        tmp = tmp.tail(1)
        tmp.reset_index(drop=True,inplace=True)
        gtex_psi_vs_expr = pd.concat([gtex_psi_vs_expr,tmp],1)
        gtex_psi_vs_expr.rename(columns={'jonckheere_pval_'+gene_name:'jonckheere_pval_global'},inplace=True)
        gtex_psi_vs_expr['gene_name']=gene_name
        gtex_psi_vs_expr['isoforms'] = ','.join(input_df['long_id'])
        gtex_psi_vs_expr.to_csv(
            directory+'/gtex_psi_vs_expr.tsv', 
                              sep=str('\t'),index=None,header=True,encoding='utf-8')
        print(directory+' done psi vs mRNA and protein expression in tissues (2)')
    except Exception as e:
        print(e)
        print(directory+' failed psi vs mRNA and protein expression in tissues (2)')
        if ('gtex_psi_vs_expr' in locals() or 'gtex_psi_vs_expr' in globals()) and len(gtex_psi_vs_expr)>0:
            tmp = pd.DataFrame([[1,1,1,1]],columns=['jonckheere_pval_psi','jonckheere_pval_global','jonckheere_pval_local_expr','jonckheere_pval_protein'])
            gtex_psi_vs_expr = pd.concat([gtex_psi_vs_expr,tmp],1)
            gtex_psi_vs_expr['gene_name']=gene_name
            gtex_psi_vs_expr['isoforms'] = ','.join(input_df['long_id'])
            gtex_psi_vs_expr.to_csv(
                directory+'/gtex_psi_vs_expr.tsv', 
                                  sep=str('\t'),index=None,header=True,encoding='utf-8')            
    if known_regulators is not None:
        ### ПРОВЕРЯЕМ ИЗВЕСТНЫЕ РЕГУЛЯТОРЫ
        try:
            # убираем авторегуляцию
            known_regulators = known_regulators.loc[known_regulators['regulator']!=gene_name]

            # проверяем известные регуляторы из литературы - сплайсинг
            experiments_on_regulators_rMATS = get_rMATS_data_from_regulators_experiments(input_df,AS_type,
                                                                                   SRA_directory='/home/magmir/TASS/DATA/SRA/',
                                                                                   RBP_shRNA_KD_dir='/home/magmir/TASS/DATA/RBP_shRNA_KD_hg19/',
                                                                                  rscript_path = '/mnt/lustre/tools/R/R-3.6.3/bin/',
                                                                                  regulators=list(known_regulators['regulator']))
            if len(experiments_on_regulators_rMATS)>0:
                grouped_exper_results_rMATS = group_experiments_results(experiments_on_regulators_rMATS,'delta_psi')
                grouped_exper_results_rMATS.rename(columns={'num_of_expers':'num_of_expers_delta_psi',
                                                            'num_of_sign_expers':'num_of_sign_expers_delta_psi',
                                                            'qval':'qval_delta_psi',
                                                            'consistent_sign':'consistent_sign_delta_psi'},
                                                   inplace=True)
                print(directory+' done splicing in experiments for known regulators (3)')
                # проверяем известные регуляторы из литературы - экспрессия
                experiments_on_regulators_deseq = get_deseq_data_from_regulators_experiments(gene_name,
                                                                                             SRA_directory='/home/magmir/TASS/DATA/SRA/',
                                                                                             RBP_shRNA_KD_dir='/home/magmir/TASS/DATA/RBP_shRNA_KD_hg19/',
                                                                                             rscript_path = '/mnt/lustre/tools/R/R-3.6.3/bin/',
                                                                                             regulators = list(known_regulators['regulator']))
                experiments_on_regulators_deseq.sort_values(['regulator','qval','log2FC'],ascending=[True,True,False],inplace=True)
                grouped_exper_results_deseq = group_experiments_results(experiments_on_regulators_deseq,'log2FC')
                grouped_exper_results_deseq.rename(columns={'num_of_expers':'num_of_expers_log2FC',
                                                            'num_of_sign_expers':'num_of_sign_expers_log2FC',
                                                            'qval':'qval_log2FC',
                                                            'consistent_sign':'consistent_sign_log2FC'},
                                                   inplace=True)
                
                known_reg_result = pd.merge(grouped_exper_results_rMATS,grouped_exper_results_deseq,
                                            how='outer',on='regulator')
                print(directory+' done expression in experiments for known regulators (4)')
                known_reg_result_details = pd.merge(experiments_on_regulators_rMATS.rename(columns={'pval':'pval_delta_psi','qval':'qval_delta_psi','lFDR':'lFDR_delta_psi'}),
                                                    experiments_on_regulators_deseq.rename(columns={'pval':'pval_log2FC','qval':'qval_log2FC','lFDR':'lFDR_log2FC'}),
                                                    how='inner',on=['regulator','Database','cell_line','experiment'])
                known_reg_result_details.sort_values(['regulator','Database','cell_line','experiment','qval_delta_psi','qval_log2FC'],inplace=True)
                known_reg_result_details[['qval_delta_psi','qval_log2FC','lFDR_delta_psi','lFDR_log2FC','pval_delta_psi','pval_log2FC','log2FC']]=np.round(known_reg_result_details[['qval_delta_psi','qval_log2FC','lFDR_delta_psi','lFDR_log2FC','pval_delta_psi','pval_log2FC','log2FC']],3)
                known_reg_result_details['consistent_expr_and_spl'] = ((known_reg_result_details['delta_psi']*known_reg_result_details['log2FC']<0).astype('int')*2-1)*(known_reg_result_details['qval_delta_psi']<0.05).astype('int')*(known_reg_result_details['qval_log2FC']<0.05).astype('int')
                known_reg_result_details = known_reg_result_details[['regulator','Database','cell_line','experiment','consistent_expr_and_spl','delta_psi','qval_delta_psi','lFDR_delta_psi','log2FC','qval_log2FC','lFDR_log2FC']]
                known_reg_result_details.to_csv(
                directory+'/known_regulators_detailed.tsv', 
                                  sep=str('\t'),index=None,header=True,encoding='utf-8')

            else:
                known_reg_result=pd.DataFrame(columns=['regulator', 'num_of_expers_delta_psi', 'num_of_sign_expers_delta_psi',
       'qval_delta_psi', 'median_abs_delta_psi', 'delta_psi_max',
       'delta_psi_min', 'consistent_sign_delta_psi', 'num_of_expers_log2FC',
       'num_of_sign_expers_log2FC', 'qval_log2FC', 'median_abs_log2FC',
       'log2FC_max', 'log2FC_min', 'consistent_sign_log2FC'])
            # теперь смотрим корреляции в GTEx 
            res = get_correlations_of_psi_with_regulators_gtex(junction_count_data,list(known_regulators['regulator']),partial_cors=True,glasso=False,backward_association=False,rscript_path='/mnt/lustre/tools/R/R-3.6.3/bin/',glasso_reg_param=1)
            junction_count_data = res[0]
            regulators_gtex = res[1]
            regulators_gtex_pres = regulators_gtex.copy()
            for feature in ['log2FC', 'pval_clust_adj', 'rho', 'pval_rho_adj', 'pcor', 'pval_pcor_adj']:
                if feature in regulators_gtex_pres.columns:
                    regulators_gtex_pres[feature] = np.round(regulators_gtex_pres[feature].astype('float'),3)

            regulators_gtex_pres = regulators_gtex_pres[['regulator', 'log2FC', 'pval_clust_adj', 'rho', 'pval_rho_adj', 'pcor',
                   'pval_pcor_adj']]
            print(directory+' done psi vs expression in gtex samples for known regulators (5)')

            known_reg_result = pd.merge(known_reg_result,regulators_gtex_pres.rename(columns={'log2FC':'log2FC_GTEx',
                                                                                      'pval_clust_adj':'pval_clust_adj_GTEx',
                                                                                      'rho':'rho_GTEx',
                                                                                     'pval_rho_adj':'pval_rho_adj_GTEx',
                                                                                     'pcor':'pcor_GTEx',
                                                                                     'pval_pcor_adj':'pval_pcor_adj_GTEx'}),how='outer',on='regulator')
            known_reg_result = pd.merge(known_reg_result,known_regulators,how='right',on=['regulator'])
            known_reg_result['exper_validated_splicing'] = ((known_reg_result['regulation_mode']=='NMD-promoting').astype('int')*2-1)*((known_reg_result['delta_psi_max']<0).astype('int')*2-1)*known_reg_result['consistent_sign_delta_psi']*((known_reg_result['qval_delta_psi']<0.05)|(known_reg_result['num_of_expers_delta_psi']>2)).astype('int')
            known_reg_result['exper_validated_splicing'] = known_reg_result['exper_validated_splicing'].fillna(0).astype('int')

            known_reg_result['exper_validated_expression'] = ((known_reg_result['regulation_mode']=='NMD-promoting').astype('int')*2-1)*((known_reg_result['log2FC_max']>0).astype('int')*2-1)*known_reg_result['consistent_sign_log2FC']*((known_reg_result['qval_log2FC']<0.05)|(known_reg_result['num_of_expers_log2FC']>2)).astype('int')
            known_reg_result['exper_validated_expression'] = known_reg_result['exper_validated_expression'].fillna(0).astype('int')

            known_reg_result['validated_GTEx_splicing'] = ((known_reg_result['regulation_mode']=='NMD-promoting').astype('int')*2-1)*((known_reg_result['log2FC_GTEx']>0).astype('int')*2-1)*(known_reg_result['pval_clust_adj_GTEx']<0.05).astype('int')

            known_reg_result[['num_of_expers_delta_psi','num_of_sign_expers_delta_psi','consistent_sign_delta_psi',
                             'num_of_expers_log2FC','num_of_sign_expers_log2FC','consistent_sign_log2FC']] = known_reg_result[['num_of_expers_delta_psi','num_of_sign_expers_delta_psi','consistent_sign_delta_psi',
                             'num_of_expers_log2FC','num_of_sign_expers_log2FC','consistent_sign_log2FC']].fillna(0).astype('int')
            known_reg_result[['qval_delta_psi','median_abs_delta_psi','delta_psi_max','delta_psi_min',
                              'qval_log2FC','median_abs_log2FC','log2FC_max','log2FC_min',
                              'log2FC_GTEx','pval_clust_adj_GTEx','rho_GTEx','pval_rho_adj_GTEx',
                              'pcor_GTEx','pval_pcor_adj_GTEx']] = known_reg_result[['qval_delta_psi','median_abs_delta_psi','delta_psi_max','delta_psi_min',
                              'qval_log2FC','median_abs_log2FC','log2FC_max','log2FC_min',
                              'log2FC_GTEx','pval_clust_adj_GTEx','rho_GTEx','pval_rho_adj_GTEx',
                              'pcor_GTEx','pval_pcor_adj_GTEx']].fillna(0)
            known_reg_result.rename(columns={'num_of_expers_delta_psi':'num_of_expers'},inplace=True)
            known_reg_result = known_reg_result[['regulator','regulation_mode','num_of_expers',
                                                 'exper_validated_splicing','exper_validated_expression','validated_GTEx_splicing',                                                'num_of_sign_expers_delta_psi','qval_delta_psi','delta_psi_max','delta_psi_min',
                                                 'num_of_sign_expers_log2FC','qval_log2FC','log2FC_max','log2FC_min',
                                                'log2FC_GTEx','pval_clust_adj_GTEx','rho_GTEx', 'pval_rho_adj_GTEx', 'pcor_GTEx','pval_pcor_adj_GTEx']]
            known_reg_result.sort_values(['exper_validated_splicing','exper_validated_expression','validated_GTEx_splicing',
                                          'qval_delta_psi','qval_log2FC','pval_clust_adj_GTEx'],ascending=[False,False,False,True,True,True],inplace=True)
            known_reg_result.to_csv(
                directory+'/known_regulators_grouped.tsv', 
                                  sep=str('\t'),index=None,header=True,encoding='utf-8')
        except Exception as e:
            print(e)
            print(directory+' failed analysis of known regulators')
    if find_novel_regulators:
        ### ПРОВЕРЯЕМ НОВЫЕ РЕГУЛЯТОРЫ
        try:
            # проверяем новые регуляторы - сплайсинг
            experiments_on_regulators_rMATS = get_rMATS_data_from_regulators_experiments(input_df,AS_type,
                                                                                   SRA_directory='/home/magmir/TASS/DATA/SRA/',
                                                                                   RBP_shRNA_KD_dir='/home/magmir/TASS/DATA/RBP_shRNA_KD_hg19/',
                                                                                  rscript_path = '/mnt/lustre/tools/R/R-3.6.3/bin/',
                                                                                  regulators=None,exclude=[gene_name])
            grouped_exper_results_rMATS = group_experiments_results(experiments_on_regulators_rMATS,'delta_psi')
            grouped_exper_results_rMATS.rename(columns={'num_of_expers':'num_of_expers_delta_psi',
                                                        'num_of_sign_expers':'num_of_sign_expers_delta_psi',
                                                        'qval':'qval_delta_psi',
                                                        'consistent_sign':'consistent_sign_delta_psi'},
                                               inplace=True)
            # проверяем новые регуляторы - экспрессия
            experiments_on_regulators_deseq = get_deseq_data_from_regulators_experiments(gene_name,
                                                                                         SRA_directory='/home/magmir/TASS/DATA/SRA/',
                                                                                         RBP_shRNA_KD_dir='/home/magmir/TASS/DATA/RBP_shRNA_KD_hg19/',
                                                                                         rscript_path = '/mnt/lustre/tools/R/R-3.6.3/bin/',
                                                                                         regulators = None,exclude=[gene_name])
            grouped_exper_results_deseq = group_experiments_results(experiments_on_regulators_deseq,'log2FC')
            grouped_exper_results_deseq.rename(columns={'num_of_expers':'num_of_expers_log2FC',
                                                        'num_of_sign_expers':'num_of_sign_expers_log2FC',
                                                        'qval':'qval_log2FC',
                                                        'consistent_sign':'consistent_sign_log2FC'},
                                               inplace=True)

            novel_reg_result = pd.merge(grouped_exper_results_rMATS,grouped_exper_results_deseq,
                                        how='outer',on='regulator')
            print(directory+' done expression in experiments for novel regulators (7)')
            novel_reg_result_details = pd.merge(experiments_on_regulators_rMATS.rename(columns={'pval':'pval_delta_psi','qval':'qval_delta_psi','lFDR':'lFDR_delta_psi'}),
                                                experiments_on_regulators_deseq.rename(columns={'pval':'pval_log2FC','qval':'qval_log2FC','lFDR':'lFDR_log2FC'}),
                                                how='inner',on=['regulator','Database','cell_line','experiment'])
            novel_reg_result_details.sort_values(['regulator','Database','cell_line','experiment','qval_delta_psi','qval_log2FC'],inplace=True)
            novel_reg_result_details[['qval_delta_psi','qval_log2FC','lFDR_delta_psi','lFDR_log2FC','pval_delta_psi','pval_log2FC','log2FC']]=np.round(novel_reg_result_details[['qval_delta_psi','qval_log2FC','lFDR_delta_psi','lFDR_log2FC','pval_delta_psi','pval_log2FC','log2FC']],3)
            novel_reg_result_details['consistent_expr_and_spl'] = ((novel_reg_result_details['delta_psi']*novel_reg_result_details['log2FC']<0).astype('int')*2-1)*(novel_reg_result_details['qval_delta_psi']<0.05).astype('int')*(novel_reg_result_details['qval_log2FC']<0.05).astype('int')
            
            novel_reg_result_details = novel_reg_result_details[['regulator','Database','cell_line','experiment','consistent_expr_and_spl','delta_psi','qval_delta_psi','lFDR_delta_psi','log2FC','qval_log2FC','lFDR_log2FC']]
            novel_reg_result_details.to_csv(
            directory+'/novel_regulators_detailed.tsv', 
                              sep=str('\t'),index=None,header=True,encoding='utf-8')

            # теперь смотрим корреляции в GTEx для всех регуляторов из списка
            candidate_regulators = list(novel_reg_result['regulator'].unique())
    #         list(novel_reg_result.loc[((novel_reg_result['qval_delta_psi']<0.05)&(novel_reg_result['consistent_sign_delta_psi']==1))|(
    #             (novel_reg_result['qval_log2FC']<0.05)&(novel_reg_result['consistent_sign_log2FC']==1))]['regulator'].unique())
            res = get_correlations_of_psi_with_regulators_gtex(junction_count_data,candidate_regulators,partial_cors=True,glasso=False,backward_association=False,
                                                               rscript_path='/mnt/lustre/tools/R/R-3.6.3/bin/',
                                                               glasso_reg_param=1)
            junction_count_data = res[0]
            regulators_gtex = res[1]

            regulators_gtex_pres = regulators_gtex.copy()
            for feature in ['log2FC', 'pval_clust_adj', 'rho', 'pval_rho_adj', 'pcor', 'pval_pcor_adj']:
                if feature in regulators_gtex_pres.columns:
                    regulators_gtex_pres[feature] = np.round(regulators_gtex_pres[feature].astype('float'),3)

            regulators_gtex_pres = regulators_gtex_pres[['regulator', 'log2FC', 'pval_clust_adj', 'rho', 'pval_rho_adj', 'pcor',
                   'pval_pcor_adj']]
            print(directory+' done psi vs expression in gtex samples for novel regulators (8)')

            novel_reg_result = pd.merge(novel_reg_result,regulators_gtex_pres.rename(columns={'log2FC':'log2FC_GTEx',
                                                                                      'pval_clust_adj':'pval_clust_adj_GTEx',
                                                                                      'rho':'rho_GTEx',
                                                                                     'pval_rho_adj':'pval_rho_adj_GTEx',
                                                                                     'pcor':'pcor_GTEx',
                                                                                     'pval_pcor_adj':'pval_pcor_adj_GTEx'}),how='outer',on='regulator')
            novel_reg_result[['num_of_expers_delta_psi','num_of_sign_expers_delta_psi','consistent_sign_delta_psi',
                             'num_of_expers_log2FC','num_of_sign_expers_log2FC','consistent_sign_log2FC']] = novel_reg_result[['num_of_expers_delta_psi','num_of_sign_expers_delta_psi','consistent_sign_delta_psi',
                             'num_of_expers_log2FC','num_of_sign_expers_log2FC','consistent_sign_log2FC']].fillna(0).astype('int')
            novel_reg_result[['qval_delta_psi','median_abs_delta_psi','delta_psi_max','delta_psi_min',
                              'qval_log2FC','median_abs_log2FC','log2FC_max','log2FC_min',
                              'log2FC_GTEx','pval_clust_adj_GTEx','rho_GTEx','pval_rho_adj_GTEx',
                              'pcor_GTEx','pval_pcor_adj_GTEx']] = novel_reg_result[['qval_delta_psi','median_abs_delta_psi','delta_psi_max','delta_psi_min',
                              'qval_log2FC','median_abs_log2FC','log2FC_max','log2FC_min',
                              'log2FC_GTEx','pval_clust_adj_GTEx','rho_GTEx','pval_rho_adj_GTEx',
                              'pcor_GTEx','pval_pcor_adj_GTEx']].fillna(0)
            novel_reg_result['delta_psi_cons'] = novel_reg_result.apply(lambda x:get_cons_sign(x,'delta_psi_max','delta_psi_min'),1)
            novel_reg_result['log2FC_cons'] = novel_reg_result.apply(lambda x:get_cons_sign(x,'log2FC_max','log2FC_min'),1)
            novel_reg_result['consistent_expr_and_spl_in_experiments'] = ((novel_reg_result['delta_psi_cons']*novel_reg_result['log2FC_cons']<0).astype('int')*2-1)*((novel_reg_result['qval_delta_psi']<0.05)|(novel_reg_result['num_of_expers_delta_psi']>2)).astype('int')*((novel_reg_result['qval_log2FC']<0.05)|(novel_reg_result['num_of_expers_log2FC']>2)).astype('int')
            
            novel_reg_result['consistent_spl_in_experiments_and_gtex'] = ((novel_reg_result['delta_psi_cons']*novel_reg_result['log2FC_GTEx']<0).astype('int')*2-1)*((novel_reg_result['qval_delta_psi']<0.05)|(novel_reg_result['num_of_expers_delta_psi']>2)).astype('int')*(novel_reg_result['pval_clust_adj_GTEx']<0.05).astype('int')
            novel_reg_result['predicted_mode'] = novel_reg_result.apply(lambda x: get_predicted_mode(x),1)
            novel_reg_result.rename(columns={'num_of_expers_delta_psi':'num_of_expers'},inplace=True)
            novel_reg_result = novel_reg_result[['regulator','predicted_mode','num_of_expers','consistent_expr_and_spl_in_experiments','consistent_spl_in_experiments_and_gtex',                                                'num_of_sign_expers_delta_psi','qval_delta_psi','delta_psi_max','delta_psi_min',
                                                 'num_of_sign_expers_log2FC',
                                                 'qval_log2FC','log2FC_max','log2FC_min',
                                                'log2FC_GTEx','pval_clust_adj_GTEx','rho_GTEx', 'pval_rho_adj_GTEx', 'pcor_GTEx','pval_pcor_adj_GTEx']]
            novel_reg_result.sort_values(['consistent_expr_and_spl_in_experiments','consistent_spl_in_experiments_and_gtex', 'num_of_sign_expers_delta_psi','num_of_sign_expers_log2FC','qval_delta_psi','qval_log2FC','pval_clust_adj_GTEx'],
                                         ascending=[False,False,False,False,True,True,True],inplace=True)
            novel_reg_result.to_csv(
                directory+'/novel_regulators_grouped.tsv', 
                                  sep=str('\t'),index=None,header=True,encoding='utf-8')
            print(directory+' done analysis of novel regulators (6)')
        except Exception as e:
            print(e)
            print(directory+' failed analysis of novel regulators (6)')

def get_predicted_mode(x):
    if x['delta_psi_cons']<0:
        return 'NMD-promoting'
    elif x['delta_psi_cons']>0:
        return 'NMD-inhibiting'
    else:
        return ''
            
def get_cons_sign(x,value1,value2):
    if x[value1]>0 and x[value2]>0:
        return 1
    elif x[value1]<0 and x[value2]<0:
        return -1
    else:
        return 0
    
def get_tissue(x,sign_tissues):
    if x['smtsd'] in sign_tissues:
        return x['smtsd']
    else:
        return 'other tissues'
    
from statsmodels.stats import proportion as smprop

def get_group_percentages(df, x_group, hue, sum_by=None):
    if sum_by is None:
        grouped = df.groupby([hue], sort=False)
        counts = grouped[x_group].value_counts(sort=False)
        counts = pd.DataFrame(counts).rename(columns = {x_group : 'count'}).reset_index()

        counts_sum = counts[[hue,'count']].groupby(hue).sum()
        counts_sum.reset_index(inplace=True)
        counts_sum.rename(columns={'count' : "countgroup"},inplace=True)

        data = pd.merge(counts,counts_sum,how='left',left_on=hue,right_on=hue)
        data['percentage'] = data['count']/data['countgroup']*100        
    else:
        grouped = df[[x_group,hue,sum_by]].groupby([hue, x_group], sort=False).agg(np.sum)
        grouped.reset_index(inplace=True)

        counts = grouped[[hue,sum_by]].groupby(hue).sum()
        counts.reset_index(inplace=True)
        counts = counts.rename(columns={sum_by : sum_by+"_sumgroup"})

        data = pd.merge(grouped,counts,how='left',left_on=hue,right_on=hue)
        data['percentage'] = data[sum_by]/data[sum_by+'_sumgroup']*100
        
    return data.sort_values(by=[hue,x_group])

def add_errorbars(ax_item,x,y,yerr):
    yerr_0 = [0]*len(yerr)
    yerr = [yerr_0,yerr]
    (_, caps, _) = ax_item.errorbar(x = x,
                                y = y,
                                yerr=yerr,
                                capsize=3,
                                fmt='none',
                                    color='black')
    for cap in caps:
        cap.set_markeredgewidth(1)

def get_proportion_ci(data_percentage, data_observcount,z):
    return 100*z*((data_percentage/100)*(1-data_percentage/100)/(data_observcount))**0.5

def percentage_plot(dataset, x_group, hue, axes_pos=0, legend_loc='upper right', cleanlook=False, alpha=0.05, 
                    palette='hls', dodge=0.52, xlabel='',ylabel='',legend_title='',exclude_x_group='',point_scale=0,
                   edgecolor='black',linewidth=1):    

    data=get_group_percentages(dataset, x_group, hue)
    
    if exclude_x_group!='':
        data = data.loc[~data[x_group].isin(exclude_x_group)]
    
    palette_n = len(data[[hue]].drop_duplicates())
    
    if axes_pos!=0:
        ax0 = sns.pointplot(x=x_group, scale=point_scale, y="percentage", hue=hue, data=data,ax=axes_pos,
                         palette=sns.color_palette(palette, palette_n),
#                          order = order,
                            dodge=dodge, join=False)
        ax0 = sns.barplot(x=x_group, y="percentage", hue=hue, data=data,ax=axes_pos,
                 palette=sns.color_palette(palette, palette_n),linewidth=linewidth,edgecolor=edgecolor,
#                          order = order
                 )
    else:
        ax0 = sns.pointplot(x=x_group, scale=point_scale, y="percentage", hue=hue, data=data,
                         palette=sns.color_palette(palette, palette_n),
#                          order = order,
                            dodge=dodge, join=False)
        ax0 = sns.barplot(x=x_group, y="percentage", hue=hue, data=data,
                 palette=sns.color_palette(palette, palette_n),linewidth=linewidth,edgecolor=edgecolor,
#                          order = order
                 )    
    x_coords = []
    y_coords = []
    for point_pair in ax0.collections:
        for x, y in point_pair.get_offsets():
            x_coords.append(x)
            y_coords.append(y)    
    try:
        add_errorbars(ax0, x=x_coords, y=y_coords, 
                      yerr =(data.apply(lambda x: smprop.proportion_confint(count=x['count'], 
                                                                            nobs=x['countgroup'],
                                                                            alpha=alpha,method='normal')[1],1)-data['percentage']/100)*100)
    except Exception as e: print(e)

    if legend_loc==None:
        legend = ax0.legend()
        legend.remove()
    elif legend_loc=='outside':
        leg_handles = ax0.get_legend_handles_labels()[0][len(data[hue].drop_duplicates()):]
        ax0.legend(leg_handles, list(data[hue].drop_duplicates().sort_values()),bbox_to_anchor=(1.05, 1), loc=2, title=legend_title, borderaxespad=0.)
    else:
        leg_handles = ax0.get_legend_handles_labels()[0][len(data[hue].drop_duplicates()):]
        ax0.legend(leg_handles, list(data[hue].drop_duplicates().sort_values()), title=legend_title,loc=legend_loc)

    if cleanlook:
        ax0.set(ylabel='', xlabel='')
        ax0.legend_.remove()
    if ylabel!='':
        ax0.set(ylabel=ylabel)
    if xlabel!='':
        ax0.set(xlabel=xlabel)
    return ax0

# оцениваем тканеспецифичность 04.2022
def get_ts_res(nmd_sample_data,tissue_column='smtsd'):
    if str(type(nmd_sample_data))!="<class 'pandas.core.frame.DataFrame'>":
        nmd_sample_data = pd.concat(nmd_sample_data)
    nmd_sample_data['index_event'] = nmd_sample_data['gene_name']+'_'+nmd_sample_data['AS_event_position']
    nmd_sample_data = nmd_sample_data.drop(['gene_name','AS_event_position'],1)

    a = []
    i=0
    index_events = list(nmd_sample_data['index_event'].unique())
    for index_event in index_events:
        tmp = nmd_sample_data.loc[nmd_sample_data['index_event']==index_event]
        for feature in ['psi','global_expr','local_expr']:
            feature_med = tmp[feature].median()
            for tissue in list(tmp[tissue_column].unique()):
                tmp1 = tmp.loc[tmp[tissue_column]==tissue]
                feature_tissue = tmp1[feature].median()
                x = len(tmp1.loc[(tmp1[feature]>feature_med)])
                n = len(tmp1)
                pval = stats.binom_test(x,n,alternative='two-sided')
                a.append([index_event,feature,tissue,feature_med,feature_tissue,pval])
        if i%20==0:
            print(str(i)+' out of '+str(len(index_events)))
        i=i+1

    ts_res = pd.DataFrame(a,columns=['index_event','feature',tissue_column,'feature_med','feature_'+tissue_column,'pval'])

    index_events = list(nmd_sample_data['index_event'].unique())
    a = []
    for index_event in index_events:
        tmp1 = pd.DataFrame(nmd_sample_data[tissue_column].unique(),columns=[tissue_column])
        tmp1['index_event'] = index_event
        for feature in ['psi','global_expr','local_expr']:
            tmp2 = ts_res.loc[(ts_res['feature']==feature)&(ts_res['index_event']==index_event)]
            tmp2['fdr'] = multipletests(tmp2['pval'],method = 'fdr_bh')[1]
            tmp1 = pd.merge(tmp1,
                            tmp2.rename(columns={'feature_med':feature+'_med','feature_'+tissue_column:feature,
                                                'fdr':feature+'_fdr'}).drop(['feature','pval'],1),
                            how='left',on=['index_event',tissue_column])
        a.append(tmp1)

    ts_res = pd.concat(a).reset_index(drop=True)
    return ts_res

def get_proteomicsDB_analysis(nmd_sample,NMD_cols,can_cols,all_ProtDB_tissues_x_smtsd):

    summary,nmd_sample_data = get_GTEX_summary_pooled(nmd_sample,NMD_cols,can_cols,True)
    nmd_sample_data = pd.concat(nmd_sample_data).reset_index(drop=True)
    nmd_sample_data_shortened = pd.merge(nmd_sample_data,all_ProtDB_tissues_x_smtsd.loc[all_ProtDB_tissues_x_smtsd['smtsd']!='-'],
                                         how='inner',on=['smtsd'])

    a = []
    l = list(nmd_sample_data_shortened['gene_name'].unique())
    i = 0
    for gene_name in l:
        protdf = get_proteomicsDB_expr_data(gene_name)
        if protdf is not None:
            protdf['gene_name'] = gene_name
            a.append(protdf)
        if i%50==0:
            print(str(i)+' out of '+str(len(l)))
        i=i+1
    protdf = pd.concat(a).reset_index(drop=True)
    nmd_sample_data_shortened = pd.merge(nmd_sample_data_shortened,
                                         protdf[['gene_name','TISSUE_NAME']].drop_duplicates(),
                                         how='inner',on=['gene_name','TISSUE_NAME'])

    ts_res = get_ts_res(nmd_sample_data_shortened,'TISSUE_NAME')
    ts_res = ts_res.dropna().sort_values(['index_event','psi']).reset_index(drop=True)
    ts_res['gene_name'] = ts_res['index_event'].str.split('_',expand=True)[0]
    ts_res[['index_event','gene_name','TISSUE_NAME']].to_csv(
                '/home/magmir/TASS/NMD_regulation/temp/index_event_tissue_order.tsv', 
                              sep=str('\t'),index=None,header=True,encoding='utf-8')
    protdf.to_csv(
                '/home/magmir/TASS/NMD_regulation/temp/prot_expression.tsv', 
                              sep=str('\t'),index=None,header=True,encoding='utf-8')

    rscript_path='/mnt/lustre/tools/R/R-3.6.3/bin/'

    command = rscript_path+'Rscript /home/magmir/TASS/NMD_regulation/calculate_jonckheere_test_bulk.r '+\
    '/home/magmir/TASS/NMD_regulation/temp/index_event_tissue_order.tsv '+\
    '/home/magmir/TASS/NMD_regulation/temp/prot_expression.tsv '+\
    '/home/magmir/TASS/NMD_regulation/temp/jonckheere_test_bulk_res.tsv '+\
    'decreasing '+\
    '/home/magmir/libs/R'
    os.system(command)
    jonckheere_pvals = pd.read_csv('/home/magmir/TASS/NMD_regulation/temp/jonckheere_test_bulk_res.tsv',
                                  delimiter="\t",index_col=None,header=0)
    jonckheere_pvals.columns = ['index_event','pval']
    
    return ts_res, protdf, jonckheere_pvals

def get_gtex_expression_of_regulators(regulators):

    sql = """SELECT gene_name, gene_counts FROM gene_counts_hg19 
            WHERE gene_name IN ("""+"'"+"','".join(regulators)+"'"+""")
            """
    dat_g = pd.read_sql_query(sql, conn)
    expr = dat_g['gene_counts'].astype('str').str[1:-1].str.split(', ',expand=True).transpose()
    expr.columns = list(dat_g['gene_name'])
    expr = expr.astype('int')
    expr['sample_id'] = expr.index+1
    del dat_g

    Exclude_tissues = ['Cells - Transformed fibroblasts','Cells - EBV-transformed lymphocytes','Testis']

    sample_metadata = pd.read_csv('/home/magmir/TASS/GTEX/sample_metadata.tsv',delimiter="\t",
                                   index_col=None,header=0)
    sample_metadata = sample_metadata.loc[(sample_metadata['sf']>0)&(
        sample_metadata['sf_global']>0)&(
        ~sample_metadata['smtsd'].isin(Exclude_tissues))]

    expr = pd.merge(expr,sample_metadata[['sample_id','smtsd','sf_global']],how='left',on='sample_id')

    for gene_name in regulators:
        if gene_name in list(expr.columns):
            expr[gene_name] = np.log2((1/expr['sf_global'])*(expr[gene_name]+8))
            expr[gene_name] = expr[gene_name]-expr[gene_name].median()
    return expr