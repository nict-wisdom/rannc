#!/usr/bin/env bash

MPI_OPTS=${MPI_OPTS:-}

export PYTEST=${PYTEST:-pytest}

function get_port_unused_random {
   (netstat --listening --all --tcp --numeric |
    sed '1,2d; s/[^[:space:]]*[[:space:]]*[^[:space:]]*[[:space:]]*[^[:space:]]*[[:space:]]*[^[:space:]]*:\([0-9]*\)[[:space:]]*.*/\1/g' |
    sort -n | uniq; seq 1 60000; seq 1 65535
    ) | sort -n | uniq -u | shuf -n 1
}

PARTITION_NUMS=(
    "1" "2" "4"
)

## test_simple
for PART in "${PARTITION_NUMS[@]}" ; do
  ENVS="-x RANNC_SHOW_CONFIG_ITEMS=true"
  ENVS+=" -x RANNC_PARTITION_NUM=${PART}"

  mpirun -np 4 ${MPI_OPTS} \
    -x PYTEST \
    ${ENVS} \
    ./ompi_helper \
    "$(uname -n)" "$(get_port_unused_random)" \
    test_simple.py \
    --batch-size 64
done

#pytest --co test_simple_amp.py | grep "Function" | while read -r LINE
#do
#  if [[ ${LINE} =~ (test_[_0-9a-z]+) ]]; then
#    TEST_NAME=${BASH_REMATCH[1]}
#    echo $TEST_NAME
#
#    for PART in "${PARTITION_NUMS[@]}" ; do
#      ENVS="-x RANNC_SHOW_CONFIG_ITEMS=true"
#      ENVS+=" -x CUBLAS_WORKSPACE_CONFIG=:4096:8"
#      ENVS+=" -x RANNC_PARTITION_NUM=${PART}"
#      mpirun -np 4 ${MPI_OPTS} \
#        -x PYTEST \
#        ${ENVS} \
#        ./ompi_helper \
#        "$(uname -n)" "$(get_port_unused_random)" \
#        test_simple_amp.py::${TEST_NAME} \
#        --batch-size 64
#    done
#
#  fi
#done


# test_bn
ENVS="-x RANNC_SHOW_CONFIG_ITEMS=true"
ENVS+=" -x RANNC_PARTITION_NUM=1"
ENVS+=" -x RANNC_PIPELINE_NUM=1"
mpirun -np 1 ${MPI_OPTS} \
  -x PYTEST \
  ${ENVS} \
  ./ompi_helper \
  "$(uname -n)" "$(get_port_unused_random)" \
  test_bn.py \
  --batch-size 64

# test_native
#if ninja --version 1> /dev/null 2> /dev/null ; then
#  for PART in "${PARTITION_NUMS[@]}" ; do
#
#    ENVS="-x RANNC_SHOW_CONFIG_ITEMS=true"
#    ENVS+=" -x RANNC_PARTITION_NUM=${PART}"
#
#    mpirun ${MPI_OPTS} \
#      -x PYTEST \
#      ${ENVS} \
#      ./ompi_helper \
#      "$(uname -n)" "$(get_port_unused_random)" \
#      test_native.py \
#      --batch-size 64
#
#  done
#else
#  echo  "No ninja found. Skipping test of native call."  1>&2
#fi
