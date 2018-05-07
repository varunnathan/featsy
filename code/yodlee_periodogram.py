from scipy import signal
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from featsy.registry import feat

# yodlee variables
FEATS = ['balance.weekly.amount|sum',
        'balance.monthly.amount|sum',
        'transaction.ttype.weekly.amount|sum',
        'transaction.ttype.weekly.amount|count',
        'transaction.ttype.monthly.amount|sum',
        'transaction.ttype.monthly.amount|count']

def relevanttsi(tsi,asof):
    #asof = asof_dict[userid]
    asof = pd.to_datetime(asof)
    end_date = asof
    start_date = asof-relativedelta(years=1)
    mask = (tsi.index > start_date) & (tsi.index  <= end_date)
    tsi_new = tsi.loc[mask]
    return tsi_new

def freq_feats(X):
    f, Pxx_spec = signal.periodogram(X, scaling='spectrum')
    max_power = Pxx_spec.max()
    max_power_index = np.where(Pxx_spec==max_power)[0]
    top_freq = f[max_power_index]
    time_period = 1/top_freq
    return time_period, max_power

@feat(apply=FEATS)
def periodogram(series, asof):
    # prepare training dataset
    series = 1.*series/100 # converting cents to dollar amounts
    tsi = relevanttsi(series,asof)
    tsi= tsi.fillna(method='ffill')
    tsi= tsi.fillna(method='bfill')

    if len(tsi)>=4:
        time_period,max_power = freq_feats(tsi)
        time_period = time_period[0]
        result = [time_period,len(tsi),time_period/len(tsi),max_power]

        yield 'time_period',result[0]
        yield 'tslength',result[1]
        yield 'timeperiodbytslength',result[2]
        yield 'maxpower',result[3]
