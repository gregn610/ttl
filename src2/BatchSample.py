import hashlib
import io
import calendar


import numpy as np
import pandas as pd

from Consts import CONST_EMPTY


class BatchSample(object):
    """
    Parse a log file or buffer into pandas.DataFrames.


    self.dfX - the X matrix of data features
    self.dfy - the Yhat predicted batch end time
    self.watershed - the minimum event time subtracted from each event as part of regularization
    """

    def __init__(self):

        self._dfI                     = None
        self.CONST_COLNAME_PREFIX     = '0000__C'
        self.event_time_col           = None
        self.event_label_col          = None
        self._feature_padding_columns = []
        self.dfX                      = None
        self.dfy                      = None
        self.filepath_or_buffer       = None
        self.source_was_buffer        = None

    def process_file(self, filepath_or_buffer, event_time_col_index, event_label_col_index,
                     pandas_reader='CSV', **X_pd_kwargs):
            # Overrides
            X_pd_kwargs['header']           = None
            X_pd_kwargs['prefix']           = self.CONST_COLNAME_PREFIX
            X_pd_kwargs['index_col']        = False
            X_pd_kwargs['skip_blank_lines'] = False  # Otherwise mask indexes get funky

            X_pd_kwargs['parse_dates'] = list(set(X_pd_kwargs.get('parse_dates', []) + [event_time_col_index]))

            if pandas_reader.upper() == 'CSV':
                X_pd_kwargs['skiprows'] = self._get_skiprows(filepath_or_buffer, event_time_col_index, **X_pd_kwargs)
                dfX = pd.read_csv(filepath_or_buffer, **X_pd_kwargs)
            else:
                raise 'pandas_reader=%s not yet implemented.' % pandas_reader

            self.event_time_col = dfX.columns[event_time_col_index]
            self.event_label_col = '0000__label'  # dfX.columns[event_label_col_index]
            ## Add the event hashes
            # Whatever happened to cause the log line to be written is an event.
            # An event can happen more than once per batch but should be fairly unique per batch and repeat over days.
            #
            # It might make sense to allow some tokenizing / preprocessing functionality here or
            # to pass in some hints and manipulation instructions
            dfX[self.event_label_col] = dfX.apply(lambda row: hashlib.sha1(
                row[dfX.columns[event_label_col_index]].encode('UTF-8')
            ).hexdigest(), axis=1)

            self.datetime_cols = list(dfX.select_dtypes(include=['datetime64', ]).columns.values)
            # Regularize times by finding minimum and using that as an watershed to subtract
            self.dt_watersheds = {}
            for idx, col in enumerate(self.datetime_cols):
                self.dt_watersheds[col] = dfX.min()[col]
                dfX[col + '_watershedded'] = dfX[col].apply(
                    lambda row: (row - self.dt_watersheds[col]) / np.timedelta64(1, 'm')
                )

            # Last event is the crossing of the finish line
            maxX = dfX.max()[self.event_time_col]
            dfy = pd.DataFrame({'predicted_time': pd.Series([maxX, ], dtype='datetime64[ns]')})
            dfy['finish_watershedded'] = dfy['predicted_time'].apply(lambda row: (
                                                                     row - self.dt_watersheds[
                                                                         self.event_time_col]) / np.timedelta64(1, 'm')
                                                                     )

            self.dfX = dfX
            self.dfy = dfy
            self.source_was_buffer = isinstance(filepath_or_buffer, io.IOBase)
            self.filepath_or_buffer = filepath_or_buffer # Leave this last as proxy for assessing success


                # The columns from the file aren't watershedded and so aren't X features

        #        self.dfX.non_feature_columns = X_names
        # Want to preserve order too
        #        self.dfX.feature_columns = list(x for x in dfX.columns if x not in set(self.dfX.non_feature_columns))

    def _get_skiprows(self, filepath_or_buffer, event_time_col_index, **X_pd_kwargs):
        # It's imperative that the event_time column parse to a valid datetime
        # So create a mask of all lines that don't coerce and then skip loading them.

        mask_kwargs = X_pd_kwargs.copy()
        mask_kwargs['usecols'] = [event_time_col_index]
        dfMask = pd.read_csv(filepath_or_buffer, **mask_kwargs)
        dfMask = pd.isnull(pd.to_datetime(dfMask.iloc[:, 0], errors='coerce'))

        # Start stream again
        if isinstance(filepath_or_buffer, io.IOBase):
            filepath_or_buffer.seek(0)

        skiprows = dfMask.index[dfMask]
        return skiprows

    # A few what's what column descriptors.
    # Cant be sets because then the order of features could change, invalidating the learning
    def _get_bool_cols(self):
        return (self.dfX.select_dtypes(include=['bool']).columns.values).tolist()

    def _get_number_cols(self):
        return self.dfX.select_dtypes(include=[np.number]).columns.values.tolist()

    def _get_datetime_cols(self):
        return (self.dfX.select_dtypes(include=['datetime64', ]).columns.values).tolist()

    def _get_timedelta_cols(self):
        return (self.dfX.select_dtypes(include=['timedelta', ]).columns.values).tolist()

    def get_dfX_feature_cols(self):
        """
        datetime_cols is excluded because they're now all timedeltas and booleans
        Use list comprehension for preserve order
        """
        return [col for col in self.dfX.columns if col in (
            self._get_bool_cols() + self._get_number_cols() + self._get_timedelta_cols())]

    def pad_feature_columns(self, complete_features):
        """
        Not every BatchSample will have every feature. But keras wants everything nice and square so
        keep a record of column names that should be padded out.
        """
        self._feature_padding_columns = [m for m in complete_features if m not in self.dfX.columns]

    def get_non_feature_cols(self):
        return [col for col in self.dfX.columns if col not in self.get_dfX_feature_cols()]

    def get_event_labels(self):
        """
        """
        return self.dfX[self.event_label_col].unique()


    def _get_dfI(self, dfX=None, refresh=False):
        """
        Returns a dataframe of the dfX data, pivoted, filled down, datetime cols enriched with weekday one-hots,
        and complete with _feature_padding_columns of CONST_EMPTY

        :param dfX:
        :param refresh:
        :return: pd.DataFrame
        """
        if self._dfI is None or refresh == True:
            if dfX is None:
                dfX = self.dfX

            dfI = dfX.pivot(columns=self.event_label_col,
                            index=self.event_time_col,
                            values=self.event_time_col + '_watershedded'
                            )
            dfI.fillna(method='ffill', inplace=True)
            dfI.reset_index(inplace=True)

            # Add 7 day of week Boolean feature columns (mon-sun) for each datetime column
            for col in self.dt_watersheds.keys():
                for idx, dabbr in enumerate(calendar.day_abbr):
                    colname = col + '_weekday_' + dabbr.lower()
                    dfI[colname] = dfI[col].map(lambda x: x.weekday() == idx)

            # These were lost in the pivot
            missing_cols = [m for m in self.get_dfX_feature_cols() if m not in dfI.columns]
            for m in missing_cols:
                dfI[m] = self.dfX[m]

            # Make sure all required cols are present
            missing_cols = [m for m in self._feature_padding_columns if m not in dfI.columns]
            for m in missing_cols:
                dfI[m] = pd.Series([CONST_EMPTY, ] * len(dfI.index))

            dfI.fillna(CONST_EMPTY, inplace=True)
            dfI.drop(self.event_time_col, inplace=True, axis=1)

            # Do columns.tolist() to make sure that the features alway line up in the same order
            self._dfI = dfI[sorted(dfI.columns.tolist())]

        return self._dfI

    def get_raw_non_features_X(self):
        # Why is this soooo slow ?
        # 1 loop, best of 3: 51.9 s per loop
        nfc = self.get_non_feature_cols()
        return self.dfX[nfc].values

    def get_dfI_values(self):
        return self._get_dfI(self.dfX).astype(float).values

    def get_raw_y(self):
        return self.dfy.values

    def regularizedToDateTime(self, colname, regularized):
        return self.dt_watersheds[colname] + pd.Timedelta(minutes=(regularized))

