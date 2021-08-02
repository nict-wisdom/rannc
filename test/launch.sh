#!/usr/bin/env bash

#MPI_OPTS=${MPI_OPTS:-"-x LD_PRELOAD=/lustre1/home/mtnk/work/nccl_v2.9.9-1/build/lib/libnccl.so --mca pml ucx --mca btl ^vader,tcp,openib --mca coll ^hcoll"}
MPI_OPTS=${MPI_OPTS:-"--mca pml ucx --mca btl ^vader,tcp,openib --mca coll ^hcoll"}

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

  mpirun --tag-output -np 4 ${MPI_OPTS} \
    -x PYTEST \
    ${ENVS} \
    ./ompi_helper \
    "$(uname -n)" "$(get_port_unused_random)" \
    test_simple.py \
    --batch-size 64

  sleep 10
done

## test_half
#HALF_TESTS=(
#    "test_half_amp"
#    "test_half_loss_amp"
#    "test_half_loss_amp_layernorm"
#    "test_half_loss_amp_save"
#)
#for PART in "${PARTITION_NUMS[@]}" ; do
#  ENVS="-x RANNC_SHOW_CONFIG_ITEMS=true"
#  ENVS+=" -x RANNC_PARTITION_NUM=${PART}"
#
#  for TEST in "${HALF_TESTS[@]}" ; do
#    mpirun --tag-output -np 4 ${MPI_OPTS} \
#      -x PYTEST \
#      ${ENVS} \
#      ./ompi_helper \
#      "$(uname -n)" "$(get_port_unused_random)" \
#      ${TEST}.py \
#      --batch-size 64
#    sleep 10
#  done
#done

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
#ENVS="-x RANNC_SHOW_CONFIG_ITEMS=true"
#ENVS+=" -x RANNC_PARTITION_NUM=1"
#ENVS+=" -x RANNC_PIPELINE_NUM=1"
#mpirun -np 1 ${MPI_OPTS} \
#  -x PYTEST \
#  ${ENVS} \
#  ./ompi_helper \
#  "$(uname -n)" "$(get_port_unused_random)" \
#  test_bn.py \
#  --batch-size 64

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
