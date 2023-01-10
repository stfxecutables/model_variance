./run_python.sh "-m pytest --rootdir="$(readlink -f test)" --ignore=$(readlink -f test/test_splitting.py) --ignore=ABIDE-eig --ignore=ec-analyses-2021 $(readlink -f test)"
