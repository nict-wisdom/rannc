#!/bin/bash

# qsub parameters
NODES=${NODES:-1}
NGPUS=${NGPUS:-4}
NP=$((NODES*NGPUS))
LOCAL_PROCS=${NGPUS}

CONDA_ENV=${CONDA_ENV:-rannc}
QUEUE=${QUEUE:-}

# script parameters
BATCH_SIZE=${BATCH_SIZE:-64}
PARTITION_NUM=${PARTITION_NUM:-1}
TIMESTAMP=${TIMESTAMP:-$(date +"%Y%m%d%H%M%S")}

PARAMS="NP${NP}_MP${PARTITION_NUM}_BS${BATCH_SIZE}"

# qsub log location
BASE_DIR=$(cd $(dirname $0); pwd)
TASK_NAME=rannc_tests_${PARAMS}
LOG_DIR=${BASE_DIR}/pbs_logs
mkdir -p ${LOG_DIR}
STD_LOG=${LOG_DIR}/${TASK_NAME}_${TIMESTAMP}.out
ERR_LOG=${LOG_DIR}/${TASK_NAME}_${TIMESTAMP}.err

echo "Submitting ${TASK_NAME}"

ENV_VARS="NP=${NP}"
ENV_VARS+=",BATCH_SIZE=${BATCH_SIZE}"
ENV_VARS+=",PARTITION_NUM=${PARTITION_NUM}"
ENV_VARS+=",TIMESTAMP=${TIMESTAMP}"
ENV_VARS+=",CONDA_ENV=${CONDA_ENV}"

QUEUE_OPT=""
if [[ -n "${QUEUE}" ]]; then
  QUEUE_OPT="-q${QUEUE}"
fi

# submit
qsub ${QUEUE_OPT} -o ${STD_LOG} -joe \
    -l select=${NODES}:ncpus=20:ngpus=${NGPUS}:mem=32gb:mpiprocs=${LOCAL_PROCS}:ompthreads=1 \
    -l walltime=2:0:0 \
    -N ${TASK_NAME} \
    -v ${ENV_VARS} \
    ${BASE_DIR}/launch.sh
