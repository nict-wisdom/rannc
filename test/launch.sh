#!/usr/bin/env bash

HOST=$(hostname)
HOST_PREFIX=$(sed -E 's/[0-9]*$//g' <<< "$HOST")

if [[ -z "${PBS_O_WORKDIR}" ]]; then
  SCRIPT_DIR=$(cd $(dirname $0); pwd)
  cd ${SCRIPT_DIR}
else
  cd ${PBS_O_WORKDIR}
fi

ENV_FILE="./${HOST_PREFIX}.env"

# Overwrite configurations here
if [ -e ${ENV_FILE} ]; then
  echo "Loading environment configurations from ${ENV_FILE}"
  source ${ENV_FILE}
fi

export PYTEST=${PYTEST:-pytest}
MPI_OPTS=${MPI_OPTS:-}

function get_port_unused_random {
   (netstat --listening --all --tcp --numeric |
    sed '1,2d; s/[^[:space:]]*[[:space:]]*[^[:space:]]*[[:space:]]*[^[:space:]]*[[:space:]]*[^[:space:]]*:\([0-9]*\)[[:space:]]*.*/\1/g' |
    sort -n | uniq; seq 1 60000; seq 1 65535
    ) | sort -n | uniq -u | shuf -n 1
}

NP=${NP:-4}
BATCH_SIZE=${BATCH_SIZE:-32}
PARTITION_NUM=${PARTITION_NUM:-1}
RANNC_CONF_DIR=${RANNC_CONF_DIR:-${HOME}/.pyrannc}

echo "NP=${NP}"
echo "PYTHON=$(which python)"
echo "JOB_ID=${PBS_JOBID}"
echo "TORCH=$(python -c "import torch; print('torch.__version__={}'.format(torch.__version__))")"
echo "RANNC=$(python -c "import pkg_resources; print(pkg_resources.require('pyrannc'))")"
echo "CONF_DIR=${RANNC_CONF_DIR}"

ENVS="-x RANNC_SHOW_CONFIG_ITEMS=true"
ENVS+=" -x RANNC_PARTITION_NUM=${PARTITION_NUM}"
ENVS+=" -x RANNC_CONF_DIR=${RANNC_CONF_DIR}"
ENVS+=" ${MPI_ENV_OPTS}"

set -x

mpirun --tag-output -np ${NP} ${MPI_OPTS} \
  -x PYTEST \
  ${ENVS} \
  ./ompi_helper \
  "$(uname -n)" "$(get_port_unused_random)" \
  test_simple.py \
  --batch-size ${BATCH_SIZE}
