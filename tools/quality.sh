#!/usr/bin/env bash

for qc in KLMinimizationV2 SQNR MSE KLMinimization None  Percentile
do
  for qs in symmetric_with_uint8 asymmetric symmetric # symmetric_with_power2_scale
  do
      for as in symmetric_with_uint8 asymmetric symmetric #symmetric_with_power2_scale
      do
        [ "$qc" == "Percentile" ] && qc="${qc} -percentile-calibration-value=99.99  "
        profiling/test_new.sh evaluate.onnx "-quantization-calibration=${qc} -quantization-schema-constants=$qs -quantization-schema-activations=$as"
      done
  done
done