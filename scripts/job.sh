#run for permission: chmod u+x job.sh
#run for execution: ./job.sh

#!/bin/sh
#source activate opt_env

python exp_mnist.py 'True'
python exp_gisette.py 'True'
python exp_covtype.py 'True'
python exp_sido.py 'True' 1e-3
python exp_sido.py 'True' 1e-2
python exp_tstudent.py 'True' 2
python exp_tstudent.py 'True' 4
python exp_tstudent.py 'True' 1