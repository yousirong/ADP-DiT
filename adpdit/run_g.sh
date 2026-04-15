model='DiT-g/2'
params=" \
            --qk-norm \
            --model ${model} \
            --rope-img base512 \
            --rope-real \
            "

numactl --interleave=all \
deepspeed --master_port=${MASTER_PORT} adpdit/train_deepspeed.py ${params} "$@"
