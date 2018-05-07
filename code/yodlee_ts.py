import pandas as pd
import statsmodels.api as sm
from dateutil.relativedelta import relativedelta
from featsy.registry import feat
from featsy.handlers.aggregate import *
import warnings
warnings.filterwarnings("ignore")

# TIME SERIES PARAMETERS
# SARIMA (p,d,q)(P,D,Q,S) parameters for different features
pdq_monthly = {'sum_bal':[((0, 0, 0), (1, 0, 1, 1)),((0, 0, 0), (1, 1, 1, 1)),((1, 0, 0), (1, 1, 0, 1))],
                'count_credit':[((0, 0, 1), (1, 0, 0, 1)),((1, 0, 0), (0, 0, 1, 1)),((0, 1, 1), (1, 0, 0, 3))],
                'count_debit':[((0, 0, 1), (1, 0, 0, 1)),((0, 0, 0), (0, 1, 1, 1)),((0, 0, 1), (0, 1, 0, 1))],
                'sum_credit':[((1, 0, 0), (0, 0, 1, 1)),((0, 0, 0), (0, 1, 1, 1)),((1, 0, 0), (1, 1, 0, 1))],
                'sum_debit':[((1, 0, 0), (0, 0, 1, 1)),((0, 0, 0), (0, 1, 1, 1)),((1, 0, 0), (1, 1, 0, 1))]}

pdq_weekly = {'sum_bal':[((1, 0, 0), (1, 0, 1, 2)),((0, 0, 0), (1, 0, 1, 2)),((1, 0, 1), (1, 0, 1, 4))],
                'count_credit':[((1, 0, 0), (1, 1, 1, 1)),((0, 0, 0), (1, 1, 1, 2)),((1, 0, 0), (0, 0, 1, 1))],
                'count_debit':[((0, 0, 0), (1, 0, 1, 4)),((1, 0, 1), (1, 0, 0, 2)),((1, 0, 0), (0, 0, 1, 1))],
                'sum_credit':[((1, 0, 0), (1, 0, 1, 2)),((0, 0, 1), (1, 0, 1, 1)),((1, 0, 0), (0, 1, 1, 2))],
                'sum_debit':[((0, 0, 1), (1, 0, 1, 2)),((0, 0, 0), (1, 0, 1, 4)),((0, 0, 1), (1, 0, 0, 1))]}

test_size_level = {'weekly':[10,20,30],'monthly':[2,4,6]}
freq_dict = {'W': 'weekly', 'M': 'monthly'}
pdq_dict = {'W': pdq_weekly, 'M': pdq_monthly}

method_dict = {'txn':['sum', 'count'],'bal':['sum']}
ttype_dict = {'txn':['debit','credit'],'bal':['bal']}

out_feat_suff = ['pred10','pred20','pred30','avg10','avg20','avg30']

def relevanttsi(tsi,asof):
    asof = pd.to_datetime(asof)
    end_date = asof
    start_date = asof-relativedelta(years=1)
    mask = (tsi.index > start_date) & (tsi.index  <= end_date)
    tsi_new = tsi.loc[mask]
    return tsi_new

def evaluate_arima_model(X,param,param_seasonal,agg_level):

    train_size=len(X)
    train = [val for val in X]

    test_size = test_size_level[agg_level]

    model = sm.tsa.statespace.SARIMAX(train,order = param,seasonal_order = param_seasonal)
    model_fit = model.fit(disp=0)

    yhat = model_fit.predict(start=train_size, end=train_size+max(test_size)-1)
    predictions = yhat.tolist()

    pred0 = predictions[test_size[0]-1]
    pred1 = predictions[test_size[1]-1]
    pred2 = predictions[test_size[2]-1]
    avg0 = sum(predictions[0:test_size[0]])/test_size[0]
    avg1 = sum(predictions[0:test_size[1]])/test_size[1]
    avg2 = sum(predictions[0:test_size[2]])/test_size[2]

    return [pred0,pred1,pred2,avg0,avg1,avg2]

def agg_yodlee(data,freq,method,asof,ttype):
    df = data.copy()
    last_day = asof.normalize()
    df.sort_index(inplace=True)

    if ttype=='bal':
        df_filt = df
    else:
        df_filt = df[df['ttype'] == ttype]
        df_filt.reset_index(drop=True, inplace=True)

    df_filt['date'] = df_filt['date'].apply(pd.to_datetime)
    df_filt['freq_indices'] = AGGREGATIONS[freq](df_filt['date'], last_day)

    if method == 'sum':
        df_filt_agg = pd.DataFrame(df_filt.groupby('freq_indices')['amount'].sum())

    elif method == 'count':
        df_filt_agg = pd.DataFrame(df_filt.groupby('freq_indices')['amount'].count())

    return pd.Series(df_filt_agg['amount'].values,index=df_filt_agg.index)

def datatype(data):
    if 'ttype' in data.columns.values:
        dtype = 'txn'
    else:
        dtype = 'bal'
    return dtype

@feat(apply=['transaction','balance'])
def sarima(data, asof):
    data['amount'] = 1.*(data['amount']/100)
    dtype = datatype(data) # check if data is txn or bal data

    for freq in ['W', 'M']:
        for method in method_dict[dtype]:
            for ttype in ttype_dict[dtype]:

                level = freq_dict[freq]
                pdq_picked = pdq_dict[freq]
                pdq_comb = pdq_picked[method+'_'+ttype]

                tsi = agg_yodlee(data,level,method,asof,ttype)
                tsi = relevanttsi(tsi,asof)
                tsi= tsi.fillna(method='ffill')
                tsi= tsi.fillna(method='bfill')

                #check if there are still missing values
                if tsi.isnull().sum()==0:

                    if len(tsi)>=4:
                        error_attempts=0
                        for i in range(3):
                            pdq,seasonal_pdq = pdq_comb[i]
                            try:
                                #print 'running ARIMA'
                                result = evaluate_arima_model(tsi,pdq,seasonal_pdq,level)
                                break
                            except:
                                error_attempts = error_attempts+1
                                continue
                        if error_attempts==3:
                            if level == 'weekly':
                                result = [tsi[-1:][0],tsi[-1:][0],tsi[-1:][0],sum(tsi[-10:])/10,sum(tsi[-20:])/20,sum(tsi[-30:])/30]
                            else:
                                result = [tsi[-1:][0],tsi[-1:][0],tsi[-1:][0],sum(tsi[-2:])/2,sum(tsi[-4:])/4,sum(tsi[-6:])/6]

                        suff = test_size_level[level]

                        feat_num=0
                        for res in result:
                            if not pd.isnull(res):
                                yield dtype+ttype+'_A'+level+'_amtC'+method+out_feat_suff[feat_num],res
                            feat_num = feat_num + 1
