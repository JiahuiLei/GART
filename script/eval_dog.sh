dataset="dog_demo"
profile="dog"
logbase=${profile}

SEQUENCES=(
    "shiba" 
    "alaskan"  
    "corgi" 
    "french" 
    "english"    
    "pit_bull"  
    "german"  
    "hound" 
)

for seq in ${SEQUENCES[@]}; do
    python solver.py --profile ./profiles/dog/${profile}.yaml --dataset $dataset --seq ${seq} --eval_only --log_dir logs/${logbase}/seq=${seq}_prof=${profile}_data=${dataset}
done