FOR PROVER9

go to symbolic_solvers/Prover9/
run make test1
    make test2
    make test3
if there are any errors run make all, retry the test and see the specific .bin missing


EXPERIMENTS

./run_pive_ms.sh --dataset chunk1.json --output corrected_chunk1.json --batch_size 1 --sample_limit -1 --continuous True

sample_limit = -1 means full dataset
