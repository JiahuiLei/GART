dataset="instant_avatar_wild" # in ins-ava processing formart
profile="ubc_mlp"
logbase=${profile}

SEQUENCES=(
    "91+bCFG1jOS"
    "91Iegdp9HFS"
    "91fxYsir49S"
    "91QFEra7jDS"
    "91Ile3zLhMS"
    "91WvLcNpdzS"
)

for seq in ${SEQUENCES[@]}; do
    python solver.py --profile ./profiles/ubc/${profile}.yaml --dataset $dataset --seq ourpose_ubc_${seq} --logbase $logbase --fast --no_eval
done