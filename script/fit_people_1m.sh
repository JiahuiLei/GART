dataset="people_snapshot"
profile="people_1m"
logbase=${profile}


for seq in  "male-3-casual" "male-4-casual" "female-3-casual" "female-4-casual"
do
    python solver.py --profile ./profiles/people/${profile}.yaml --dataset $dataset --seq $seq --logbase $logbase --fast --no_eval
done