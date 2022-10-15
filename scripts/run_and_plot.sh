#run for permission: chmod u+x run_and_plot.sh
#run for execution: ./run_and_plot.sh

#!/bin/sh
#source activate opt_env

python exp_mnist.py 1 1 1 
python exp_gisette.py 1 1 1
python exp_covtype.py 1 1 1
python exp_sido.py 1 1 1 1e-3
python exp_sido.py 1 1 1 1e-2
python exp_tstudent.py 1 1 1 2
python exp_tstudent.py 1 1 1 4
python exp_tstudent.py 1 1 1 1