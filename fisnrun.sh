export SiamMask=$PWD

set -x PYTHONPATH $PWD $PYTHONPATH


cd $SiamMask/experiments/siammask_sharp
set -x PYTHONPATH $PWD $PYTHONPATH

python ../../tools/demo.py --resume SiamMask_VOT.pth --config config_davis.json
