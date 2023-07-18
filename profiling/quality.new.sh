#!/usr/bin/env bash

for as in asymmetric  symmetric #symmetric_with_uint8 symmetric_with_power2_scale
do
  for qs in symmetric  asymmetric # symmetric_with_uint8 symmetric_with_power2_scale
  do
     for qc in KLMinimization KLMinimizationV2 MSE SQNR  None  Percentile
      do
        [ "$qc" == "Percentile" ] && qc="${qc} -percentile-calibration-value=99.9950  "
        profiling/test.new.sh $1 "-quantization-calibration=${qc} -quantization-schema-constants=$qs -quantization-schema-activations=$as"
      done
  done
done
