#!/bin/bash
MODEL=$1
LOCAL_TEST_DIR="${MODEL}.prof"
MD5LOCAL=md5
MD5LINIX=md5sum
[ "${MODEL}" != "" ] && [ -f ${MODEL} ] && MD5_MODEL=$(${MD5LOCAL} -r ${MODEL} | cut -f1 -d ' ') || exit
mkdir -p ${LOCAL_TEST_DIR}

[ ! -f "${MD5_MODEL}" ] && cp ${MODEL} "${MD5_MODEL}.onnx"
REMOTE_TEST_DIR="/home/ubuntu/${LOCAL_TEST_DIR}"
REMOTE_BUILD_DIR="${REMOTE_TEST_DIR}/${MD5_MODEL}"
LOCAL_BUILD_DIR="${LOCAL_TEST_DIR}/${MD5_MODEL}"
mkdir -p "${LOCAL_BUILD_DIR}"

CALIBRATION_FILE_LIST=$(python -c "import onnx; model = onnx.load('${MODEL}'); print('files/0.txt' if model.graph.input[0].type.tensor_type.shape.dim[1].dim_value < model.graph.input[0].type.tensor_type.shape.dim[3].dim_value else 'files/1.txt')")

#34
#PARAMS="-m=/tmp/${TEST}/${MODEL} -aic-binary-dir=/tmp/${TEST}/bin -aic-hw -aic-num-cores=1 -mos=1 -ols=1 -batchsize=1 -quantization-precision=Int8 -quantization-precision-bias=Int32 -quantization-schema-constants=symmetric_with_uint8 -quantization-schema-activations=asymmetric -aic-enable-depth-first -size-split-granularity=2048 -allocator-dealloc-delay=9 -onnx-define-symbol=batch,1 -onnx-define-symbol=unk__80,1 -time-passes -aic-perf-warnings -aic-perf-metrics -device-id=0 -compile-only -vtcm-working-set-limit-ratio=1.0  -ddr-stats -stats-level=100"
#35
#PARAMS="-m=/tmp/${TEST}/${MODEL} -aic-binary-dir=/tmp/${TEST}/bin -aic-hw -aic-num-cores=1 -mos=1 -ols=1 -batchsize=1 -quantization-precision=Int8 -quantization-precision-bias=Int32 -quantization-schema-constants=symmetric_with_uint8 -quantization-schema-activations=asymmetric  -onnx-define-symbol=batch,1 -onnx-define-symbol=unk__80,1 -time-passes -aic-perf-warnings -aic-perf-metrics -device-id=0 -compile-only -ddr-stats -stats-level=100 -aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=5 -use-producer-dma=1"
#36
#PARAMS="-m=/tmp/${TEST}/${MODEL} -aic-binary-dir=/tmp/${TEST}/bin -aic-hw -aic-num-cores=1 -mos=1 -ols=1 -batchsize=1 -quantization-precision=Int8 -quantization-precision-bias=Int32 -quantization-schema-constants=symmetric_with_uint8 -quantization-schema-activations=asymmetric  -onnx-define-symbol=batch,1 -onnx-define-symbol=unk__80,1 -time-passes -aic-perf-warnings -aic-perf-metrics -device-id=0 -compile-only -ddr-stats -stats-level=100 -aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=4 -use-producer-dma=1"
#-compile-only

QUANTIZE="8"
CALIBRATE="1"
PROFILE="0"
NOTRANDOM="1"
CUSTOM_IO="0"
PRECISION="0"
TRACE="1"

[ "${QUANTIZE}" == "8" ] && [ "$2" != "" ] && Q_FLAGS="$2" || Q_FLAGS=""
#echo Q_FLAGS="${Q_FLAGS}"

#[ "${QUANTIZE}" == "8" ] && Q_FLAGS="-quantization-precision=Int8 -quantization-precision-bias=Int32 -quantization-schema-constants=asymmetric -quantization-schema-activations=asymmetric"
[ "${QUANTIZE}" == "8" ] && Q_FLAGS="${Q_FLAGS} -quantization-precision=Int8 -quantization-precision-bias=Int32" # -quantization-schema-constants=symmetric_with_uint8"
[ "${QUANTIZE}" == "8" ] && [ ${CALIBRATE} == "1" ] && Q_FLAGS="${Q_FLAGS} -input-list-file=/home/ubuntu/${CALIBRATION_FILE_LIST} -profiling-threads=128 -compile-threads=128"
# -quantization-calibration=SQNR -profiling-threads=128"
# -quantization-calibration=MSE -profiling-threads=128"
# -quantization-calibration=Percentile -percentile-calibration-value=99.00 -profiling-threads=128"
[ "${QUANTIZE}" == "16" ] && Q_FLAGS="-convert-to-fp16"

COMPILE_FLAGS="-m=${REMOTE_BUILD_DIR}/${MD5_MODEL}.onnx -aic-hw -aic-hw-version=2.0 -aic-num-cores=1 -mos=1 -ols=1 -batchsize=1 \
-onnx-define-symbol=batch,1 -onnx-define-symbol=unk__80,1 -time-passes -time=5 -aic-perf-warnings -aic-perf-metrics -device-id=0  \
${Q_FLAGS}\
 "
[ "${TRACE}" == "1" ] && COMPILE_FLAGS="${COMPILE_FLAGS} -ddr-stats -stats-level=100"
[ "${PROFILE}" == "0" ] && [ "${QUANTIZE}" == "8" ] && FLAGS="${FLAGS} -convert-to-quantize"

#PERF_FLAGS="-aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=4 -use-producer-dma=1 -enable-channelwise"
#PERF_FLAGS="-aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=4 -use-producer-dma=1 -enable-rowwise"
#PERF_FLAGS="-aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=4"
PERF_FLAGS="-aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=6"
#PARAMS="-m=/tmp/${MODEL} -aic-binary-dir=/tmp/${TEST}/bin -aic-hw -aic-num-cores=1 -mos=1 -ols=1 -batchsize=1 -quantization-precision=Int8 -quantization-precision-bias=Int32 -quantization-schema-constants=symmetric_with_uint8 -quantization-schema-activations=asymmetric  -onnx-define-symbol=batch,1 -onnx-define-symbol=unk__80,1 -time-passes -aic-perf-warnings -aic-perf-metrics -device-id=0 -compile-only -ddr-stats -stats-level=100 -aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=4 -use-producer-dma=1"
PRECISION_FLAGS=""

COMPILE_FLAGS="${COMPILE_FLAGS} ${PERF_FLAGS} "
COMPILE_MD5=$(echo ${COMPILE_FLAGS} | ${MD5LOCAL} ).${PRECISION}

[ "${CUSTOM_IO}" == "1" ] && DUMP_IO_FILE="${MD5_MODEL}.custom_IO_config.dump.yaml"
[ "${CUSTOM_IO}" == "1" ] && DUMP_IO_FLAGS="-dump-custom-IO-config-template=${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile/${DUMP_IO_FILE}"
[ "${CUSTOM_IO}" == "1" ] && LOAD_IO_FILE="${MD5_MODEL}.custom_IO_config.load.yaml"
[ "${CUSTOM_IO}" == "1" ] && LOAD_IO_FLAGS="-custom-IO-list-file=${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_IO_FILE}"
[ "${PROFILE}" == "1" ] && DUMP_PROFILE_FILE="${MD5_MODEL}.profile.dump.yaml"
[ "${PROFILE}" == "1" ] && DUMP_PROFILE_FLAGS="-dump-profile=${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile/${DUMP_PROFILE_FILE}" || DUMP_PROFILE_FLAGS=""
[ "${PROFILE}" == "1" ] && LOAD_PROFILE_FILE="${MD5_MODEL}.profile.load.yaml"
[ "${PROFILE}" == "1" ] && LOAD_PROFILE_FLAGS="-load-profile=${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_PROFILE_FILE}" || DUMP_PROFILE_FLAGS=""

#FLAGS="${FLAGS}  ${DUMP_PROFILE_FLAGS}"
LOAD_PRECISION_FILE="${MD5_MODEL}.precision.yaml"

#export IO_FLAGS="     ${FLAGS} ${DUMP_IO_FLAGS}"
export PROFILE_FLAGS="${COMPILE_FLAGS} ${LOAD_IO_FLAGS} ${DUMP_PROFILE_FLAGS}"
export COMPILE_FLAGS="${COMPILE_FLAGS} ${LOAD_IO_FLAGS} ${LOAD_PROFILE_FLAGS}"
COMPILE_FLAGS="${COMPILE_FLAGS} -aic-binary-dir=${REMOTE_BUILD_DIR}/${COMPILE_MD5}/binary"
echo ${COMPILE_MD5} ${COMPILE_FLAGS}

#COMP_HOST=ubuntu@blackmunk.com
#COMP_EXEC="ssh -p   2023 ${COMP_HOST}"
#COMP_COPY="scp -r -P 2023"
COMP_HOST="ubuntu@44.203.79.37"
COMP_EXEC="ssh -p 2022 ${COMP_HOST}"
COMP_COPY="scp -r -P 2022"


QAIC_HOST=ubuntu@blackmunk.com
QAIC_EXEC="ssh -p 2022 ${QAIC_HOST}"
QAIC_COPY="scp -r -P 2022"

${COMP_EXEC} "rm -rf ${REMOTE_BUILD_DIR} && mkdir -p ${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile" && \
  mkdir -p "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile"
cp "${MD5_MODEL}.onnx" "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/${MD5_MODEL}.onnx"
${COMP_COPY} "${MD5_MODEL}.onnx" "${COMP_HOST}:${REMOTE_BUILD_DIR}/${MD5_MODEL}.onnx"

[ "${CUSTOM_IO}" == "1" ] && \
  [ ! -f "${LOCAL_BUILD_DIR}/${DUMP_IO_FILE}" ] && \
  ${COMP_EXEC} "/opt/qti-aic/exec/qaic-exec ${FLAGS} ${DUMP_IO_FLAGS}" && \
  ${COMP_COPY} ${COMP_HOST}:${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile/${DUMP_IO_FILE} "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${DUMP_IO_FILE}"

[ "${CUSTOM_IO}" == "1" ] && cp "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${DUMP_IO_FILE}" "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_IO_FILE}"
[ "${CUSTOM_IO}" == "1" ] && ${COMP_COPY}  "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_IO_FILE}" "${COMP_HOST}:${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_IO_FILE}"

echo "${PROFILE_FLAGS}"
echo "${COMPILE_FLAGS}"

[ "${PROFILE}" == "1" ] && \
  [ ! -f "${LOCAL_BUILD_DIR}/${DUMP_PROFILE_FILE}" ] && \
  echo ${COMP_EXEC} "/opt/qti-aic/exec/qaic-exec ${PROFILE_FLAGS}" && \
  ${COMP_EXEC} "/opt/qti-aic/exec/qaic-exec ${PROFILE_FLAGS}" && \
    ${COMP_COPY} ${COMP_HOST}:${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile/${DUMP_PROFILE_FILE} "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${DUMP_PROFILE_FILE}"

#${COMP_COPY} ${COMP_HOST}:${REMOTE_BUILD_DIR} ${LOCAL_TEST_DIR}
[ "${PROFILE}" == "1" ] && cp "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${DUMP_PROFILE_FILE}" "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_PROFILE_FILE}"
[ "${PROFILE}" == "1" ] && ${COMP_COPY}  "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_PROFILE_FILE}" "${COMP_HOST}:${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_PROFILE_FILE}"

#echo 'FP16NodeInstanceNames: [ input.4 ]' > "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_PRECISION_FILE}" && \

[ "${PRECISION}" == "1" ] && \
 echo 'FP16NodeInstanceNames: [ "1104" ]' > "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_PRECISION_FILE}" && \
    ${COMP_COPY} "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_PRECISION_FILE}" "${COMP_HOST}:${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_PRECISION_FILE}" && \
    PRECISION_FLAGS="-node-precision-info=${REMOTE_BUILD_DIR}/${COMPILE_MD5}/compile/${LOAD_PRECISION_FILE}" || PRECISION_FLAGS=""

COMPILE_FLAGS="${COMPILE_FLAGS} ${PRECISION_FLAGS}"

echo "${COMPILE_FLAGS}" | tr ' ' '\n' | sort > "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/compiler_flags.txt"
set -x
if [ ! -f "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/binary/programqpc.bin" ]; then
  ${COMP_EXEC} "/opt/qti-aic/exec/qaic-exec ${COMPILE_FLAGS}" 2>&1 | tee "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/compile/compiler_log.txt"
  ${COMP_COPY} ${COMP_HOST}:${REMOTE_BUILD_DIR}/${COMPILE_MD5}/binary ${LOCAL_BUILD_DIR}/${COMPILE_MD5}
# [ -f "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/binary/programqpc.bin" ] && ${COMP_EXEC} "${MD5LINIX} ${REMOTE_BUILD_DIR}/${COMPILE_MD5}/binary/programqpc.bin" || exit
  ${COMP_EXEC} rm -rf "${REMOTE_TEST_DIR}"
fi
set +x
####### Compilation done
[ -f "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/profile/accuracy.txt" ] && \
echo "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/profile/accuracy.txt" already exists && \
exit

set -x
${QAIC_EXEC} "rm -rf ${REMOTE_TEST_DIR} && mkdir -p ${REMOTE_BUILD_DIR}/${COMPILE_MD5}/binary ${REMOTE_BUILD_DIR}/${COMPILE_MD5}/profile ${REMOTE_BUILD_DIR}/${COMPILE_MD5}/output"  && \
${QAIC_COPY} "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/binary/programqpc.bin" "${QAIC_HOST}:${REMOTE_BUILD_DIR}/${COMPILE_MD5}/binary/programqpc.bin" && \
[ "${TRACE}" == "1" ] && TRACE_FLAG="-v trace --aic-profiling-type trace --aic-profiling-num-samples 5 \
  --aic-profiling-start-iter 10 --aic-profiling-type stats --aic-profiling-out-dir ${REMOTE_BUILD_DIR}/${COMPILE_MD5}/profile ${TRACE_FLAG}" || TRACE_FLAG=""

[ "${NOTRANDOM}" == "1" ] && TRACE_FLAG="${TRACE_FLAG} --aic-batch-input-file-list  /home/ubuntu/files/0.txt --write-output-num-samples 500 --write-output-dir  ${REMOTE_BUILD_DIR}/${COMPILE_MD5}/output"

${QAIC_EXEC} "${MD5LINIX} ${REMOTE_BUILD_DIR}/${COMPILE_MD5}/binary/programqpc.bin && /opt/qti-aic/exec/qaic-api-test -n 500 -t ${REMOTE_BUILD_DIR}/${COMPILE_MD5}/binary ${TRACE_FLAG}" > "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/execute.log"

date && \
${QAIC_EXEC} "find ${REMOTE_BUILD_DIR}/${COMPILE_MD5}/output -type f -name 'aiccyclecounts*.bin' -delete" && \
${QAIC_EXEC} "cd ${REMOTE_BUILD_DIR}/${COMPILE_MD5} && tar  czf - ." | ( cd ${LOCAL_BUILD_DIR}/${COMPILE_MD5} && tar xf - ) && \
date

${QAIC_EXEC} rm -rf "${REMOTE_TEST_DIR}"
####### Excution done

python  tools/eval_onnx_single_core_new.py --cfg  configs/panoptic_deeplab_R50_os32_cityscapes_768x1536_local_test.yaml --opt-model "${MODEL}" --opt-result-dir "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/output" > "${LOCAL_BUILD_DIR}/${COMPILE_MD5}/profile/accuracy.txt"

