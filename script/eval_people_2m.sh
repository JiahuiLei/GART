dataset="people_snapshot"
profile="people_2m"
logbase=${profile}

for seq in  "male-3-casual" "male-4-casual" "female-3-casual" "female-4-casual"
do
    python solver.py --profile ./profiles/people/${profile}.yaml --dataset $dataset --seq $seq --eval_only --log_dir logs/${logbase}/seq=${seq}_prof=${profile}_data=${dataset}
done