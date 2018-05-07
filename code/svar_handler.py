import re
import pandas as pd
from pandas import DatetimeIndex as DTI
import numpy as np

from .handler import (DataHandler, MultiviewHandler,
                      MultiRecordHandler, MonetaryMixin)
from .dataviews import make_views
from .dataframe import get_offset
from .revenue import Revenue
from ..utilities import map_handlers, Cache, document
from ..utilities.records import make_viewkey, parse_funckey


__all__ = ['YodleeSVAR']

AGGREGATIONS = {'tx': ['weekly','monthly'],
                'bl': ['weekly','monthly']}

VIEW_FUNCTIONS = {'tx': {'weekly': ['sum','count'],'monthly': ['sum','count']},
                  'bl': {'weekly': ['sum'],'monthly': ['sum']}}

CHANNEL = {'yodlee':'yodlee','balance':'yodlee_balance'}

def make_funckeys(prefix, aggregations, funcs, AGG):
    for agg in [AGG]:
        for func in funcs[agg]:
            yield prefix + '.{}.amount|{}'.format(agg, func)


class WeeklyYodleeCustomerViews(MultiRecordHandler):
    """Compute 30 day aggregate stats for Yodlee channel for cross channel
    metrics computation."""
    name = 'weeklyyodleeviews'
    _node = 'weekly_yodlee_views'
    datatype = 'YL'
    date_columns = ['date']

    def postprocess(self):
        for col in ['txweeklysumcr', 'txweeklysumdb',
                    'txweeklycountcr', 'txweeklycountdb',
                    'blweeklysum']:
            if col not in self.data.columns:
                self.data[col] = np.nan

class MonthlyYodleeCustomerViews(MultiRecordHandler):
    """Compute 30 day aggregate stats for Yodlee channel for cross channel
    metrics computation."""
    name = 'monthlyyodleeviews'
    _node = 'monthly_yodlee_views'
    datatype = 'YL'
    date_columns = ['date']

    def postprocess(self):
        for col in ['txmonthlysumcr', 'txmonthlysumdb',
                    'txmonthlycountcr', 'txmonthlycountdb',
                    'blmonthlysum']:
            if col not in self.data.columns:
                self.data[col] = np.nan

class YodleeSVAR(MultiviewHandler):
    """Calculate kabbage debt to external revenue/cashflow features.

    Uses either rawdata or precomputed views for calculation.
    If argument data['aggregated'] is True, assumes that data is dictionary
    of precomputed views.  Otherwise, assumes rawdata.
    If using rawdata, views computed in intermediate step are also returned,
    and can be reused as precomputed views.

    Precomputed views could be used in the case that only the kabbage debt
    variables are changed but the yodlee, paypal, revenue data remains the
    same.
    """
    name = 'yodleesvar'
    _prefix = 'YS'
    _node = 'YS'
    _data_views = {}

    @classmethod
    def register_function(cls, viewkey, func, apply):
        viewkey = apply
        if cls.name not in cls._feature_functions:
            cls._feature_functions[cls.name] = {}
        ff = cls._feature_functions[cls.name]
        key = (viewkey, func.name)
        if key in ff:
            raise ValueError('`{}` is already registered for {}'
                             .format(func.name, apply))
        ff[key] = func
        views = []
        viewkey = viewkey[viewkey.find('.') + 1:]
        channelviews = viewkey.split(',')
        for view in channelviews:
            items = view.split('.')
            channel = items[0]
            viewkey = parse_funckey(items)
            views.append((channel, viewkey))
        ff[key] = (func, views)

    #def __init__(self, **data):
    def __init__(self, **data):
        self.data = data
        self.aggregated = data.get('aggregated', False)
        self.channels = ['weeklyyodleeviews','monthlyyodleeviews']
        self.channelviews = {'weeklyyodleeviews': self._calc_weekly_yodlee_customer,'monthlyyodleeviews': self._calc_monthly_yodlee_customer}

        handlers = map_handlers(DataHandler)
        if not self.aggregated:
            self.asof = None
            for channel in ['yodlee', 'yodlee_balance']:
                _ = data.get(channel)
                if _ is not None:
                    setattr(self, channel, handlers[channel](_))
                else:
                    setattr(self, channel, None)
        else:
            self.asof = data.get('asof')
            for channel in self.channelviews:
                setattr(self, channel, handlers[channel](data.get(channel)))
            self.history = handlers['history'](data.get('history'))

    def __call__(self, *args, **kwargs):
        kwargs = self._parse_kwargs(kwargs)
        feats = {}
        cache = Cache()
        prefix = self.prefix
        ff = self.feature_functions()

        asof = [kwargs['asof'], self.asof][self.aggregated]

        viewkey = make_viewkey(**kwargs)
        if not self.aggregated:
            for view, func in self.channelviews.items():
                setattr(self, view, func(viewkey))

        for k, v in ff.items():

            if not hasattr(v, 'count'):
                continue

            viewkey, name = k
            func, views = v

            _views, cols, name_str = self._get_views(kwargs, views)

            if len(_views) != len(views):
                continue

            featname = re.sub('_+', '_', prefix + name_str[0])
            func(_views, cols, featname, feats=feats,
                 cache=cache, asof=asof,
                 doc=self._doc, profile=self.profile)

        if self._doc:
            feats = document(self.name, feats)

        elif feats and not self.profile:
            if isinstance(self, MultiviewHandler):
                prefix += self.name
            for k in kwargs:
                feats['_' + prefix + k] = str(kwargs[k])

        self._cache = None
        self._doc = False
        self.profile = False

        if self.aggregated or kwargs.get('feats_only', False):
            return feats

        views = {}
        for viewname in self.channelviews:
            view = getattr(self, viewname)
            if view is None:
                continue
            data = view.data.copy()
            data['date'] = data.date.astype(str)
            if 'index' in data:
                data.drop('index', inplace=1, axis=1)
            views[viewname] = \
                data.dropna(how='all', subset=[c for c in data if
                                               c not in {"date", "index"}])\
                .to_dict(orient='records')

        return {'features': feats, 'views': views}
        #return feats

    def _get_views(self, kwargs, views):
        _views = []
        cols = []
        name_str = []
        for channel, view in views:
            kw = vars(view)
            kw.update(kwargs)
            handler = getattr(self, channel, None)
            if handler is not None:
                for _name, _view in handler.iterviews(**kw):
                    _views.append(_view)
                    name_str.append(_name)
                cols.append(view.columns)
        return _views, cols, name_str
    '''
    def _calc_paypal_customer(self, viewkey):
        views = self._calc_customer(self.paypal, self.paypal_balance, None,
                                    viewkey)
        if len(views):
            return PayPalCustomerViews(views)
    '''
    def _calc_weekly_yodlee_customer(self, viewkey):
        views = self._calc_customer(self.yodlee, self.yodlee_balance, None,
                                    viewkey,'weekly')

        if len(views):
            return WeeklyYodleeCustomerViews(views)

    def _calc_monthly_yodlee_customer(self, viewkey):
        views = self._calc_customer(self.yodlee, self.yodlee_balance, None,
                                    viewkey,'monthly')

        if len(views):
            return MonthlyYodleeCustomerViews(views)

    '''
    def _calc_revenue_customer(self, viewkey):
        views = self._calc_customer(None, None, self.revenue, viewkey)
        if len(views):
            return RevenueCustomerViews(views)
    '''
    def _calc_customer(self, txn, bal, multi, viewkey, AGG):
        """Calculates dataframe of aggregated views for a channel.
        """

        kwargs = viewkey._asdict()
        kwargs['lastndays'] = kwargs.pop('last_ndays')
        kwargs['ignoredays'] = kwargs.pop('ignore_days')

        views = pd.DataFrame()
        if multi is not None:
            # multi is used to describe the type of handler that revenue
            # handler uses
            multi_name_str = multi.name
        else:
            multi_name_str = ''

        keys = ['transaction.ttype', 'balance']
        prefixes = ['tx', 'bl']

        for handler, key, prefix in zip([txn, bal, multi], keys, prefixes):
            if handler is None:
                continue

            aggregations = AGGREGATIONS[prefix]
            functions = VIEW_FUNCTIONS[prefix]

            if handler.data is None:
                continue
            view = self._make_customer_views(handler,
                                             make_funckeys(key, aggregations,
                                                           functions,AGG),
                                             viewkey, prefix, **kwargs)
            if len(view):
                views = pd.merge(views, view, how='outer', left_index=True,
                                 right_index=True)

        views['date'] = views.index
        return views.reset_index()

    def _make_customer_views(self, handler, funckeys, viewkey, prefix,
                             **kwargs):
        """Uses a channel handler to calculate views specific to the handler.
        """

        view = handler.getdataview(**kwargs)
        view.sort(handler.sortby_columns, inplace=1)
        view.drop_duplicates(handler.unique_columns, inplace=1, take_last=1)
        if not len(view):
            return {}

        asof = kwargs.get('asof')
        ignore_days = kwargs.get('ignore_days')

        views = pd.DataFrame(index=self._get_dti(**kwargs))

        for funckey in funckeys:
            funckey = parse_funckey(funckey.split('.'))
            key = prefix + funckey.aggregate + funckey.columns[0][1]
            handler._aggregate(funckey.aggregate, view, asof, ignore_days)
            for k, v in make_views(view, prefix, viewkey, funckey).items():
                if 'typGcr' in k:
                    views[key + 'cr'] = v
                elif 'typGdb' in k:
                    views[key + 'db'] = v
                else:
                    views[key] = v
        return views

    def _get_dti(self, **kwargs):
        asof = kwargs.get('asof')
        lastndays = kwargs.get('lastndays')
        offset = kwargs.get('offset')
        if lastndays is None:
            lastndays = 365
        if offset is None:
            offset = '0d'
        start = pd.to_datetime(asof) - pd.DateOffset(days=lastndays)
        end = pd.to_datetime(asof) + get_offset(offset)
        dates = DTI(pd.date_range(start, end))
        return dates.normalize()

    def _parse_kwargs(self, kwargs):
        for kw in kwargs:
            if kw not in {'asof', 'last_ndays', 'offset', 'doc', 'profile',
                          'ignore_days'}:
                raise ValueError('unknown keyword argument: {}'.format(kw))
        if 'asof' not in kwargs:
            kwargs['asof'] = pd.to_datetime('now')
        else:
            kwargs['asof'] = pd.to_datetime(kwargs['asof'])
        self._doc = kwargs.pop('doc', False)
        self.profile = kwargs.pop('profile', False)

        self._cache = {}
        return kwargs
