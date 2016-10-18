import pandas as pd
import numpy as np
from BatchSample import BatchSample

class DebugBatchSample(BatchSample):
    """
    This is superfunk way to cast to subclass. Thanks: http://stackoverflow.com/a/597324/266387

    Bits of useful debug info that don't belong in a regular BatchSample
    """
    def __init__(self, *args, **kwargs):
        super(DebugBatchSample, self).__init__()
        if self.filepath_or_buffer is not None: # That a proxy for the BatchSample having loaded a file
            self._conversion_from_BatchSample()

    def _conversion_from_BatchSample(self):
        self.debug_filepath_or_buffer = self.filepath_or_buffer.replace('.log', '.debug')

        self.debug_pd_kwargs = {'names' : ['__dbg_realtime_finish', ],
                                'parse_dates' : [0,],
                               }

        self.debug_dfy = pd.read_csv(self.debug_filepath_or_buffer, **self.debug_pd_kwargs)
        self.debug_dfX = pd.DataFrame()
        self.debug_dfX['__dbg_criticalpath_mask'] = self.dfX[self.event_label_col].str.contains('Critical path event')

        self.debug_dfy['__dbg_realtime_watershedded']= self.debug_dfy['__dbg_realtime_finish'].apply(
                            lambda row: (row - self.dt_watersheds[self.event_time_col]) / np.timedelta64(1,'m')
                        )

