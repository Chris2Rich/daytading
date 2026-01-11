# data/adapters.py
from AlgorithmImports import *

class CustomSignalData(BaseData):

    def GetSource(self, config, date, isLive):
        return SubscriptionDataSource(
            "path/to/source.csv",
            SubscriptionTransportMedium.LocalFile
        )

    def Reader(self, config, line, date, isLive):
        data = CustomSignalData()
        data.Symbol = config.Symbol
        data.Time = date
        data.Value = float(line)
        return data
