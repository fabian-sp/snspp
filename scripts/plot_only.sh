#run for permission: chmod u+x plot_only.sh
#run for execution: ./plot_only.sh

#!/bin/sh
#source activate opt_env

python exp_mnist.py 1 0 1 
python exp_gisette.py 1 0 1
python exp_covtype.py 1 0 1
python exp_sido.py 1 0 1 1e-2
python exp_tstudent.py 1 0 1 1
python exp_tstudent.py 1 0 1 2
python exp_tstudent.py 1 0 1 4
python exp_sido_reg.py 1 0 1 