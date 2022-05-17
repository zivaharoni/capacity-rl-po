MAX_SCRIPTS=100
MIN_SCRIPTS=25
SLEEPTIME=60
GPUS=(0 1 2 3 4 5 6 7)
NUM_GPU=${#GPUS[@]}
JOBS0=$(pgrep python -a |   wc -l )
echo "current python processes on server: $JOBS0"

EXP_NAME=revision2_corrected-gradient
#RUN_NAME=hmm
#TAG_NAME=sample_sweep-
#TAG_SUFFIX="2"
#MIN_EPOCHS=20
#MAX_EPOCHS=200
#AUX_DIST=uniform
#AUX_SAMPLES=5
#GAMMA=0.5
#BPTT=25
#NUM_EPOCHS=50
i=0
#OK=0
echo "`date`: starting...  "  > log.log
# 100 500 1000 5000
for CHANNEL in "ising" "trapdoor"
do
  for CARDINALITY in {2..3}
  do
#    if [ $CARDINALITY -ge 10 ]; then
#      MAX_SCRIPTS=20
#    else
#      MAX_SCRIPTS=20
#    fi
    echo $NUM_EPOCHS
    if [ "$CHANNEL" = "trapdoor" ] && [ $CARDINALITY -ge 3 ]; then
        continue
    fi
    for inst in {1..20}
    do
    JOBS=$(pgrep python -a |   wc -l )
    echo "`date`: sample size: $SAMPLE_SIZE instance: $inst epochs: $NUM_EPOCHS"  >> log.log

    CUDA_VISIBLE_DEVICES=${GPUS[$(($i % $NUM_GPU))]}, python ./example.py \
    --exp_name "$EXP_NAME"_"$CHANNEL"-"$CARDINALITY" --config ./configs/example.json --channel $CHANNEL \
    --channel_cardinality $CARDINALITY &

    sleep 0.1

    i=$(($i+1))
    echo "current python processes on server: $(( $JOBS - $JOBS0 ))"
    while [ $(( $JOBS - $JOBS0 )) -gt $MAX_SCRIPTS ];
    do
      echo "waiting...  current python processes on server: $JOBS"
      sleep $SLEEPTIME
      JOBS=$(pgrep python -a |   wc -l )
    done
    done
  done
done
echo "`date`: done!"  >> log.log
