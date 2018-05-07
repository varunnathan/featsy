import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.api import VAR, DynamicVAR
from dateutil.relativedelta import relativedelta
from ..registry import feat
from ..utilities import ewa_hl_months

# SVAR (p,s) parameters for different features
ps_monthly = {'blmonthlysum':[(6, 3),(5, 3),(3, 3)],
                'txmonthlycountcr':[(6, 2),(6, 3),(5, 3)],
                'txmonthlycountdb':[(7, 3),(6, 2),(2, 3)],
                'txmonthlysumcr':[(6, 3),(5, 3),(3, 3)],
                'txmonthlysumdb':[(6, 2),(5, 3),(3, 3)]}

ps_weekly = {'blweeklysum':[(45, 8),(43, 12),(39, 12),(1, 12)],
                'txweeklycountcr':[(45, 2),(40, 8),(39, 12),(1, 12)],
                'txweeklycountdb':[(44, 2),(41, 8),(39, 12),(1, 2)],
                'txweeklysumcr':[(45, 8),(43, 12),(39, 12),(1, 2)],
                'txweeklysumdb':[(45, 8),(44, 12),(39, 12),(1, 2)]}

test_size_level = {'weekly':[10,20,30],'monthly':[2,4,6]}

pdq_level = {'weekly':ps_weekly,'monthly':ps_monthly}
feats = {'weekly':{'bal':['blweeklysum'],'txn':['txweeklycountcr','txweeklycountdb','txweeklysumcr','txweeklysumdb']},
         'monthly':{'bal':['blmonthlysum'],'txn':['txmonthlycountcr','txmonthlycountdb','txmonthlysumcr','txmonthlysumdb']}}
pdq_iter = {'weekly':4,'monthly':3}

out_feat_suff = ['pred10','pred20','pred30','avg10','avg20','avg30']

def relevanttsi(tsi,asof):
    asof = pd.to_datetime(asof)
    end_date = asof
    start_date = asof-relativedelta(years=1)
    mask = (tsi.index > start_date) & (tsi.index  <= end_date)
    tsi_new = tsi.loc[mask]
    return tsi_new

def evaluate_svar_model(X,p,s,feat_nm,agg_level):
    feat_list = X.columns.values.tolist()
    #add seasonal variables
    decomposition = seasonal_decompose(X[feat_nm],model='additive',freq=s)
    X['seasonal']= decomposition.seasonal
    trend = decomposition.trend
    trend = trend.fillna(method='ffill') # fill missing values with previous values
    trend = trend.fillna(method='bfill') # fill first missing value with the one before it
    X['trend']= trend

    # prepare training dataset
    train_size=len(X)
    train = X

    model = VAR(train)
    model_fit = model.fit(p)

    test_size = test_size_level[agg_level]
    yhat = model_fit.forecast(train.values[:p],train_size+max(test_size)-p)
    index = feat_list.index(feat_nm) # index of feature we want to analyse

    yhat_feat = [item[index] for item in yhat] # model output relevant to that features
    predictions=yhat_feat

    pred0 = predictions[train_size-p+test_size[0]-1]
    pred1 = predictions[train_size-p+test_size[1]-1]
    pred2 = predictions[train_size-p+test_size[2]-1]
    avg0 = sum(predictions[train_size-p:train_size-p+test_size[0]])/test_size[0]
    avg1 = sum(predictions[train_size-p:train_size-p+test_size[1]])/test_size[1]
    avg2 = sum(predictions[train_size-p:train_size-p+test_size[2]])/test_size[2]

    return [pred0,pred1,pred2,avg0,avg1,avg2]

def agglevel(data):
    if 'weekly' in data.columns.values[1]:
        dtype = 'weekly'
    else:
        dtype = 'monthly'
    return dtype

@feat(apply={'yodleesvar.weeklyyodleeviews','yodleesvar.monthlyyodleeviews'})
def svar(data,asof):
    df = pd.DataFrame(data)
    level = agglevel(df)
    df['date'] = pd.to_datetime(df['date'])
    df.index = df['date']
    df.index.name = 'date'
    df.drop(['date','index'],inplace=True,axis=1)
    df = 1.*df/100
    df = df.dropna(axis=0, how='all')

    for key in ['bal','txn']:
        for col in feats[level][key]:
            pdq_comb = pdq_level[level][col]

            pref = level+'_'+key+'_'+col+'_'
            suff = test_size_level[level]

            tsi_raw = df.fillna(method='ffill')
            tsi_raw= tsi_raw.fillna(method='bfill')
            tsi = relevanttsi(tsi_raw,asof) # tsi that is within 1 year from the asof

            if tsi[col].isnull().sum()==0:
                if len(tsi)>=4:
                    comb_iter = pdq_iter[level]
                    error_attempts=0
                    for i in range(comb_iter):
                        p,s = pdq_comb[i]
                        try:
                            result = evaluate_svar_model(tsi,p,s,col,level)
                            break
                        except:
                            error_attempts = error_attempts+1
                            continue

                    if error_attempts==comb_iter:
                        if level == 'weekly':
                            result = [tsi[col][-1:][0],tsi[col][-1:][0],tsi[col][-1:][0],sum(tsi[col][-10:])/10,sum(tsi[col][-20:])/20,sum(tsi[col][-30:])/30]
                        else:
                            result = [tsi[col][-1:][0],tsi[col][-1:][0],tsi[col][-1:][0],sum(tsi[col][-2:])/2,sum(tsi[col][-4:])/4,sum(tsi[col][-6:])/6]

                    feat_num=0
                    for res in result:
                        if not pd.isnull(res):
                            yield key+col+out_feat_suff[feat_num],res
                        feat_num = feat_num + 1
