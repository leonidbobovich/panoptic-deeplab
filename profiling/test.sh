#!/bin/bash
#MODEL="optimized_step5.5.2_fine_tuned_os_11.onnx"
MODEL="optimized_step5.5.2_nearest_sim.onnx"
MODEL="optimized_step6.0_ckpt_33k.onnx"
#MODEL="mnist-1.onnx"
LOCAL_TEST_DIR="${MODEL}.prof"
MD5LOCAL=md5
MD5LINIX=md5sum
[ -f ${MODEL} ] && MD5_MODEL=$(${MD5LOCAL} -r ${MODEL} | cut -f1 -d ' ') || exit
mkdir -p ${LOCAL_TEST_DIR}

[ ! -f "${MD5_MODEL}" ] && cp ${MODEL} "${MD5_MODEL}.onnx"
REMOTE_TEST_DIR="/tmp/${LOCAL_TEST_DIR}"
REMOTE_BUILD_DIR="${REMOTE_TEST_DIR}/${MD5_MODEL}"
LOCAL_BUILD_DIR="${LOCAL_TEST_DIR}/${MD5_MODEL}"
mkdir -p "${LOCAL_BUILD_DIR}"
#34
#PARAMS="-m=/tmp/${TEST}/${MODEL} -aic-binary-dir=/tmp/${TEST}/bin -aic-hw -aic-num-cores=1 -mos=1 -ols=1 -batchsize=1 -quantization-precision=Int8 -quantization-precision-bias=Int32 -quantization-schema-constants=symmetric_with_uint8 -quantization-schema-activations=asymmetric -aic-enable-depth-first -size-split-granularity=2048 -allocator-dealloc-delay=9 -onnx-define-symbol=batch,1 -onnx-define-symbol=unk__80,1 -time-passes -aic-perf-warnings -aic-perf-metrics -device-id=0 -compile-only -vtcm-working-set-limit-ratio=1.0  -ddr-stats -stats-level=100"
#35
#PARAMS="-m=/tmp/${TEST}/${MODEL} -aic-binary-dir=/tmp/${TEST}/bin -aic-hw -aic-num-cores=1 -mos=1 -ols=1 -batchsize=1 -quantization-precision=Int8 -quantization-precision-bias=Int32 -quantization-schema-constants=symmetric_with_uint8 -quantization-schema-activations=asymmetric  -onnx-define-symbol=batch,1 -onnx-define-symbol=unk__80,1 -time-passes -aic-perf-warnings -aic-perf-metrics -device-id=0 -compile-only -ddr-stats -stats-level=100 -aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=5 -use-producer-dma=1"
#36
#PARAMS="-m=/tmp/${TEST}/${MODEL} -aic-binary-dir=/tmp/${TEST}/bin -aic-hw -aic-num-cores=1 -mos=1 -ols=1 -batchsize=1 -quantization-precision=Int8 -quantization-precision-bias=Int32 -quantization-schema-constants=symmetric_with_uint8 -quantization-schema-activations=asymmetric  -onnx-define-symbol=batch,1 -onnx-define-symbol=unk__80,1 -time-passes -aic-perf-warnings -aic-perf-metrics -device-id=0 -compile-only -ddr-stats -stats-level=100 -aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=4 -use-producer-dma=1"
#-compile-only
FLAGS="-m=${REMOTE_BUILD_DIR}/${MD5_MODEL}.onnx -aic-hw -aic-hw-version=2.0 -aic-num-cores=1 -mos=1 -ols=1 -batchsize=1 \
-quantization-precision=Int8 -quantization-precision-bias=Int32 -quantization-schema-constants=symmetric_with_uint8 -quantization-schema-activations=asymmetric \
-onnx-define-symbol=batch,1 -onnx-define-symbol=unk__80,1 -time-passes -aic-perf-warnings -aic-perf-metrics -device-id=0  \
-ddr-stats -stats-level=100"
# -convert-to-quantize \
PERF_FLAGS="-aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=4 -use-producer-dma=1 -enable-channelwise"
PERF_FLAGS="-aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=4 -use-producer-dma=1 -enable-rowwise"
PERF_FLAGS="-aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=4"
#PARAMS="-m=/tmp/${MODEL} -aic-binary-dir=/tmp/${TEST}/bin -aic-hw -aic-num-cores=1 -mos=1 -ols=1 -batchsize=1 -quantization-precision=Int8 -quantization-precision-bias=Int32 -quantization-schema-constants=symmetric_with_uint8 -quantization-schema-activations=asymmetric  -onnx-define-symbol=batch,1 -onnx-define-symbol=unk__80,1 -time-passes -aic-perf-warnings -aic-perf-metrics -device-id=0 -compile-only -ddr-stats -stats-level=100 -aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=1536 -allocator-dealloc-delay=4 -use-producer-dma=1"
PRECISION_FLAGS=""
#PRECISION_FLAGS="-node-precision-info"

#EXTRA_FLAGS=""
#[ -f "${MODEL}.profile.yaml" ] && EXTRA_FLAGS="${EXTRA_FLAGS} -load-profile=/tmp/${TEST}/${MODEL}.profile.yaml"
#[ -f "${MODEL}.precision.yaml" ] && EXTRA_FLAGS="${EXTRA_FLAGS} -node-precision-info=/tmp/${TEST}/${MODEL}.precision.yaml"
# EXTRA_FLAGS=""
LOAD_IO_FILE="${MD5_MODEL}.custom_IO_config.load.yaml"
DUMP_IO_FILE="${MD5_MODEL}.custom_IO_config.dump.yaml"
DUMP_IO_FLAGS="${FLAGS} ${PERFORMANCE_FLAGS} -dump-custom-IO-config-template=${REMOTE_BUILD_DIR}/${DUMP_IO_FILE}"
LOAD_PROFILE_FILE="${MD5_MODEL}.profile.load.yaml"
DUMP_PROFILE_FILE="${MD5_MODEL}.profile.dump.yaml"
DUMP_PROFILE_FLAGS="${FLAGS} ${PERFORMANCE_FLAGS} -dump-profile=${REMOTE_BUILD_DIR}/${DUMP_PROFILE_FILE}"

LOAD_PRECISION_FILE="${MD5_MODEL}.precision.yaml"

LOAD_FLAGS="-load-profile=${REMOTE_BUILD_DIR}/${LOAD_PROFILE_FILE} -custom-IO-list-file=${REMOTE_BUILD_DIR}/${LOAD_IO_FILE}"
COMPILE_FLAGS="${FLAGS} -compile-only ${PERF_FLAGS} ${LOAD_FLAGS} -aic-binary-dir=${REMOTE_BUILD_DIR}/bin ${PRECISION_FLAGS}"

COMP_HOST=ubuntu@blackmunk.com
COMP_EXEC="ssh -p 2023 ${COMP_HOST}"
COMP_COPY="scp -r -P 2023"

QAIC_HOST=root@blackmunk.com
QAIC_EXEC="ssh -p 2022 ${QAIC_HOST}"
QAIC_COPY="scp -r -P 2022"

${COMP_EXEC} "rm -rf ${REMOTE_BUILD_DIR} && mkdir -p ${REMOTE_BUILD_DIR}"
${COMP_COPY} "${MD5_MODEL}.onnx" "${COMP_HOST}:${REMOTE_BUILD_DIR}/${MD5_MODEL}.onnx"

[ ! -f "${LOCAL_BUILD_DIR}/${DUMP_IO_FILE}" ] && ${COMP_EXEC} "/opt/qti-aic/exec/qaic-exec ${DUMP_IO_FLAGS}" && \
  ${COMP_COPY} ${COMP_HOST}:${REMOTE_BUILD_DIR}/${DUMP_IO_FILE} "${LOCAL_BUILD_DIR}/${DUMP_IO_FILE}"
#cat "${LOCAL_BUILD_DIR}/${DUMP_IO_FILE}" | sed -e 's/float/int8/1' > "${LOCAL_BUILD_DIR}/${LOAD_IO_FILE}"
cp "${LOCAL_BUILD_DIR}/${DUMP_IO_FILE}" "${LOCAL_BUILD_DIR}/${LOAD_IO_FILE}"
#cp "${LOCAL_BUILD_DIR}/db55afcc3cbf1346d8ed41331ead33ad.custom_IO_config.leonid.yaml" "${LOCAL_BUILD_DIR}/${LOAD_IO_FILE}"
${COMP_COPY}  "${LOCAL_BUILD_DIR}/${LOAD_IO_FILE}" "${COMP_HOST}:${REMOTE_BUILD_DIR}/${LOAD_IO_FILE}"

[ ! -f "${LOCAL_BUILD_DIR}/${DUMP_PROFILE_FILE}" ] && ${COMP_EXEC} "/opt/qti-aic/exec/qaic-exec ${DUMP_PROFILE_FLAGS} -custom-IO-list-file=${REMOTE_BUILD_DIR}/${LOAD_IO_FILE}" && \
  ${COMP_COPY} ${COMP_HOST}:${REMOTE_BUILD_DIR}/${DUMP_PROFILE_FILE} "${LOCAL_BUILD_DIR}/${DUMP_PROFILE_FILE}"
${COMP_COPY} ${COMP_HOST}:${REMOTE_BUILD_DIR} ${LOCAL_TEST_DIR}
cp "${LOCAL_BUILD_DIR}/${DUMP_PROFILE_FILE}" "${LOCAL_BUILD_DIR}/${LOAD_PROFILE_FILE}"
${COMP_COPY}  "${LOCAL_BUILD_DIR}/${LOAD_PROFILE_FILE}" "${COMP_HOST}:${REMOTE_BUILD_DIR}/${LOAD_PROFILE_FILE}"

[ "${PRECISION_FLAGS}" == "-node-precision-info" ] && echo 'FP16NodeInstanceNames: [ input.4 ]' > "${LOCAL_BUILD_DIR}/${LOAD_PRECISION_FILE}" && \
    ${COMP_COPY} "${LOCAL_BUILD_DIR}/${LOAD_PRECISION_FILE}" "${COMP_HOST}:${REMOTE_BUILD_DIR}/${LOAD_PRECISION_FILE}" && \
    PRECISION_FLAGS="${PRECISION_FLAGS}=${REMOTE_BUILD_DIR}/${LOAD_PRECISION_FILE}"
echo "${COMPILE_FLAGS}" "${PRECISION_FLAGS}" | tr ' ' '\n' > "${LOCAL_BUILD_DIR}/compiler_flags.txt"

${COMP_EXEC} "/opt/qti-aic/exec/qaic-exec ${COMPILE_FLAGS}" "${PRECISION_FLAGS}"

${COMP_COPY} ${COMP_HOST}:${REMOTE_BUILD_DIR} ${LOCAL_TEST_DIR}

[ -f "${LOCAL_BUILD_DIR}/bin/programqpc.bin" ] && ${COMP_EXEC} "${MD5LINIX} ${REMOTE_BUILD_DIR}/bin/programqpc.bin" || exit
${COMP_EXEC} rm -rf "${REMOTE_BUILD_DIR}"

${QAIC_EXEC} rm -rf "${REMOTE_BUILD_DIR} && mkdir -p ${REMOTE_TEST_DIR}"  &&
${QAIC_COPY} "${LOCAL_BUILD_DIR}" "${QAIC_HOST}:${REMOTE_TEST_DIR}" &&
${QAIC_EXEC} "${MD5LINIX} ${REMOTE_BUILD_DIR}/bin/programqpc.bin &&
  /opt/qti-aic/exec/qaic-api-test -n 1000 -t ${REMOTE_BUILD_DIR}/bin &&
  /opt/qti-aic/exec/qaic-api-test -n 1000 -t ${REMOTE_BUILD_DIR}/bin \
  --aic-profiling-out-dir ${REMOTE_BUILD_DIR} -v trace --aic-profiling-type trace --aic-profiling-num-samples 5 \
  --aic-profiling-start-iter 10 --aic-profiling-type stats" &&
${QAIC_COPY} ${QAIC_HOST}:${REMOTE_BUILD_DIR} ${LOCAL_TEST_DIR} &&
# ${QAIC_EXEC} rm -rf "${REMOTE_BUILD_DIR}"

