FOR PROVER9

go to symbolic_solvers/Prover9/
run make test1
    make test2
    make test3
if there are any errors run make all, retry the test and see the specific .bin missing


EXPERIMENTS

./run_pive_ms.sh --dataset chunk2.json --output corrected_chunk2.json --batch_size 1 --sample_limit -1 --start index 0 --end_index 2 --continuous True

sample_limit = keep -1 if you want to run on intervals
start_index = starting index of interval of dataset (default 0)
end_index = ending index of interval of dataset (-1 means full dataset)


VLLM VERSION uses vllm_correction.py to have both models on vllm

./run_pive_vllm.sh --dataset chunk2.json --output corrected_chunk2.json --batch_size 1 --sample_limit -1 --start index 0 --end_index 2 --continuous True
