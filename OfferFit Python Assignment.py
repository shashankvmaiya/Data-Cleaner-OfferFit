"""
Instructions:

Fill in the methods of the DataCleaner class to produce the same printed results
as in the comments below. Good luck, and have fun!
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from typing import Dict, Any, List


class DataCleaner:
    """
    Transform a pandas df while keeping track of the history of transformations to
    allow reverting back to earlier state.
    """
    def __init__(self, df: pd.DataFrame):
        self.current = df
        self.history = [('Initial df', self.current.copy(deep=True))]
        self.stack = [('init', None)]
        
    def adjust_dtype_main(self, types: Dict[str, Any]) -> None:
        for c, t in types.items():
            try:
                self.current = self.current.astype({c:t})
            except Exception:
                print("Column '{0}' has data that cannot be converted to {1}. Skipping this column".format(c, t))
            

    def adjust_dtype(self, types: Dict[str, Any]) -> None:
        valid_types, prev_types = {}, {}
        for t in types:
            if t in self.current.columns:
                prev_types[t] = self.current[t].dtype
                valid_types[t] = types[t]
            else:
                print('{0} column not present in df. Type not adjusted'.format(t))
        
        self.adjust_dtype_main(valid_types)
        self.history.append(('Adjusted dtype using {0}'.format(types), self.current.copy(deep=True)))
        self.stack.append(('dtype', prev_types))

    def impute_missing(self, columns: List[str]) -> None:
        prev_nans = []
        for c in columns:
            if c in self.current.columns:
                indices = self.current[self.current[c].isna()].index.tolist()
                self.current[c].fillna(self.current[c].mean(), inplace=True)
                prev_nans.append((c, indices))
            else:
                print('{0} column not present in df. Imputing not done for this column'.format(c))
        
        self.history.append(('Imputed missing in {0}'.format(columns), self.current.copy(deep=True)))
        self.stack.append(('impute', prev_nans))

    def revert(self, steps_back: int = 1) -> None:
        ops, steps = [], 0
        for i in range(steps_back):
            if self.stack[-1][0]!='init':
                op, arg = self.stack.pop()
                if op=='dtype':
                    ops.append(op)
                    steps+=1
                    self.adjust_dtype_main(arg)
                elif op=='impute':
                    ops.append(op)
                    steps+=1
                    for c, indices in arg:
                        self.current.loc[self.current.index.isin(indices), c] = np.nan
            else:
                print('Reverted back to the original df. No further changes to revert')
                break
        self.history.append(('Reverted {0} steps & {1} operations'.format(steps, ops), self.current.copy(deep=True)))

    def save(self, path: str) -> None:
        self.current.to_pickle('{0}.pkl'.format(path))

    @staticmethod
    def load(path: str) -> DataCleaner:
        df = pd.read_pickle('{0}.pkl'.format(path))
        return DataCleaner(df)

transactions = pd.DataFrame(
    {
        "customer_id": [10, 10, 13, 10, 11, 11, 10],
        "amount": [1.00, 1.31, 20.5, 0.5, 0.2, 0.2, np.nan],
        "timestamp": [
            "2020-10-08 11:32:01",
            "2020-10-08 13:45:00",
            "2020-10-07 05:10:30",
            "2020-10-08 12:30:00",
            "2020-10-07 01:29:33",
            "2020-10-08 13:45:00",
            "2020-10-09 02:05:21",
        ]
    }
)

transactions_dc = DataCleaner(transactions)

print(f"Current dataframe:\n{transactions_dc.current}")

# Current dataframe:
#    customer_id  amount            timestamp
# 0           10    1.00  2020-10-08 11:32:01
# 1           10    1.31  2020-10-08 13:45:00
# 2           13   20.50  2020-10-07 05:10:30
# 3           10    0.50  2020-10-08 12:30:00
# 4           11    0.20  2020-10-07 01:29:33
# 5           11    0.20  2020-10-08 13:45:00
# 6           10     NaN  2020-10-09 02:05:21

print(f"Current dtypes:\n{transactions_dc.current.dtypes}")

# Initial dtypes:
# customer_id      int64
# amount         float64
# timestamp       object
# dtype: object

transactions_dc.adjust_dtype({"timestamp": np.datetime64})

print(f"Changed dtypes to:\n{transactions_dc.current.dtypes}")

# Changed dtypes to:
# customer_id             int64
# amount                float64
# timestamp      datetime64[ns]

transactions_dc.impute_missing(columns=["amount"])

print(f"Imputed missing as overall mean:\n{transactions_dc.current}")

# Imputed missing as mean:
#    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21

print(f"History of changes:\n{transactions_dc.history}")

# ** Any coherent structure with history of changes **
# E.g., here's one possibility

# History of changes:
# [('Initial df',    customer_id  amount            timestamp
# 0           10    1.00  2020-10-08 11:32:01
# 1           10    1.31  2020-10-08 13:45:00
# 2           13   20.50  2020-10-07 05:10:30
# 3           10    0.50  2020-10-08 12:30:00
# 4           11    0.20  2020-10-07 01:29:33
# 5           11    0.20  2020-10-08 13:45:00
# 6           10     NaN  2020-10-09 02:05:21), ("Adjusted dtypes using {'timestamp': <class 'numpy.datetime64'>}",    customer_id  amount           timestamp
# 0           10    1.00 2020-10-08 11:32:01
# 1           10    1.31 2020-10-08 13:45:00
# 2           13   20.50 2020-10-07 05:10:30
# 3           10    0.50 2020-10-08 12:30:00
# 4           11    0.20 2020-10-07 01:29:33
# 5           11    0.20 2020-10-08 13:45:00
# 6           10     NaN 2020-10-09 02:05:21), ("Imputed missing in ['amount']",    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21)]

transactions_dc.save("transactions")
loaded_dc = DataCleaner.load("transactions")
print(f"Loaded DataCleaner current df:\n{loaded_dc.current}")

# Loaded DataCleaner current df:
#    customer_id     amount           timestamp
# 0           10   1.000000 2020-10-08 11:32:01
# 1           10   1.310000 2020-10-08 13:45:00
# 2           13  20.500000 2020-10-07 05:10:30
# 3           10   0.500000 2020-10-08 12:30:00
# 4           11   0.200000 2020-10-07 01:29:33
# 5           11   0.200000 2020-10-08 13:45:00
# 6           10   3.951667 2020-10-09 02:05:21

transactions_dc.revert()
print(f"Reverting missing value imputation:\n{transactions_dc.current}")

# Reverting missing value imputation:
#    customer_id  amount           timestamp
# 0           10    1.00 2020-10-08 11:32:01
# 1           10    1.31 2020-10-08 13:45:00
# 2           13   20.50 2020-10-07 05:10:30
# 3           10    0.50 2020-10-08 12:30:00
# 4           11    0.20 2020-10-07 01:29:33
# 5           11    0.20 2020-10-08 13:45:00
# 6           10     NaN 2020-10-09 02:05:21
