 #!/bin/sh
./ttc.py preprocess modeldata.h5 ../data/population_v2.1/*.log
./ttc.py train --reset model.ttc modeldata.h5
./ttc.py predict model.ttc ../data/population_v2.1/2007-11-11.log

