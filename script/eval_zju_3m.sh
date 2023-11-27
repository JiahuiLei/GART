dataset="zju"
profile="zju_3m"
logbase=${profile}

for seq in  "my_377" "my_386" "my_387" "my_392" "my_393"  "my_394"
do
    python solver.py --profile ./profiles/zju/${profile}.yaml --dataset $dataset --seq $seq --eval_only --log_dir logs/${logbase}/seq=${seq}_prof=${profile}_data=${dataset}
done