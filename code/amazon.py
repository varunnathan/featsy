from .handler import OrderHandler

__all__ = ['AmazonOrders']


class AmazonOrders(OrderHandler):

    name = 'amazon'
    _prefix = 'AMZ'

    column_aliases = {
        'provideraccountsurrogatekey': 'accid',
        'amazonorderid': 'ordid',
        'orderamount': 'amount',
        'purchasedate': 'date',
        'numberofitemsshipped': 'numshipped',
        'numberofitemsunshipped': 'numunshipped',
        'shippingcity': 'shipcity',
        'shippingcountrycode': 'shipcountry',
        'shippingcounty': 'shipcounty',
        'shippingname': 'shipname',
        'shippingphone': 'shipphone',
        'shippingpostalcode': 'shipzip',
        'shippingstateorregion': 'shipstate',
        'shipservicelevel': 'shipservlevel',
        'shipmentservicelevelcategory': 'shipservcategory',
    }

    mapreduce_columns = [
        'BornOnDate',
        'AmazonOrderID',
        'BuyerEmail',
        'BuyerName',
        'CreatedBefore',
        'CbaDisplayableShippingLabel',
        'EarliestDeliveryDate',
        'EarliestShipDate',
        'FulfillmentChannel',
        'IsBusinessOrder',
        'OrderAmount',
        'OrderStatus',
        'PurchaseDate',
        'PaymentMethod',
        'LastUpdateDate',
        'LatestDeliveryDate',
        'LatestShipDate',
        'OrderChannel',
        'OrderType',
        'NumberOfItemsShipped',
        'NumberOfItemsUnshipped',
        'ShippingCity',
        'ShippingCountryCode',
        'ShippingCounty',
        'ShippingName',
        'ShippingPhone',
        'ShippingPostalCode',
        'ShippingStateOrRegion',
        'ShipServiceLevel',
        'ShipmentServiceLevelCategory',
        'ProviderAccountSurrogateKey',
        'RecordOrder',
        'MapOrder'
        ]

    offline_columns = ['bornondate']

    sortby_columns = ['bornondate', 'date', 'maporder']

    unique_columns = ['accid', 'ordid']

    required_columns = unique_columns + ['date', 'amount']

    date_columns = ['bornondate', 'earliestshipdate', 'lastupdatedate',
                    'latestshipdate', 'date', 'createdbefore',
                    'earliestdeliverydate', 'latestdeliverydate']

    column_checks = {
        'amount': ['nonnegative'],
        'numshipped': ['nonnegative', 'integer'],
        'numunshipped': ['nonnegative', 'integer'],
        'orderstatus': [('takesvalues', ('pending', 'unshipped',
                                         'partiallyshipped', 'shipped',
                                         'canceled'))]
    }

    def preprocess(self):
        """Preprocess data after columns are normalized."""

        # Remove orders from the data where amount column is null
        self.data = self.data.loc[self.data['amount'].notnull()].copy()

        # clean orderstatus column, by mapping old-values to new-values
        mapping = {'0': 'pending', '1': 'unshipped', '2': 'partiallyshipped',
                   '3': 'shipped', '4': 'canceled',  # canceled not cancelled
                   0: 'pending', 1: 'unshipped', 2: 'partiallyshipped',
                   3: 'shipped', 4: 'canceled'}  # duplicated numeric keys
        get_mapping = lambda x: mapping.get(x, x)
        self.data['orderstatus'] = self.data.orderstatus.apply(get_mapping)

        # clean paymentmethod column, by mapping old-values to new-values
        mapping = {'2': 'Other', 2: 'Other', '0': '0', 0: '0', 'COD': 'COD',
                   'Other': 'Other'}
        self.data['paymentmethod'] = self.data.paymentmethod.apply(mapping.get)

        # clean fulfillmentchannel column
        mapping = {'0': 'Other', 'MFN': 'MFN', 'AFN': 'AFN'}
        get_mapping = lambda x: mapping.get(x, x)
        self.data['fulfillmentchannel'] = self.data[
         'fulfillmentchannel'].apply(get_mapping)

        # isbusinessorder
        self.data['isbusinessorder'] = self.data['isbusinessorder'].astype(
         float)

        # promise_cats
        date_cats = [('earliest{}date'.format(x), 'latest{}date'.format(x), x)
                     for x in ['ship', 'delivery']]
        for esd, lsd, name in date_cats:
            mask = ((self.data['lastupdatedate'].notnull()) &
                    (self.data[esd].notnull()) & (self.data[lsd].notnull()))
            mask1 = self.data['lastupdatedate'] <= self.data[esd]
            mask2 = ((self.data['lastupdatedate'] > self.data[esd]) &
                     (self.data['lastupdatedate'] <= self.data[lsd]))
            mask3 = self.data['lastupdatedate'] > self.data[lsd]
            self.data.loc[(mask & mask1), 'promise_cats_{}'.format(name)] = 'advance'
            self.data.loc[(mask & mask2), 'promise_cats_{}'.format(name)] = 'ontarget'
            self.data.loc[(mask & mask3), 'promise_cats_{}'.format(name)] = 'delayed'

    def postprocess(self):
        """Postprocess data after checks and conversions are complete."""

        # Convert column values to lowercase
        for col in ['orderstatus', 'paymentmethod', 'shipcity', 'shipcountry',
                    'shipservcategory', 'shipservlevel', 'shipstate',
                    'fulfillmentchannel']:
            self.data[col] = self.data[col].str.lower()
