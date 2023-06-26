#!/usr/bin/env bash

for qc in  KLMinimization # KLMinimizationV2 MSE SQNR None  Percentile
do
  for qs in symmetric # symmetric_with_uint8 asymmetric # symmetric_with_power2_scale
  do
      for as in asymmetric # symmetric_with_uint8  symmetric #symmetric_with_power2_scale
      do
        [ "$qc" == "Percentile" ] && qc="${qc} -percentile-calibration-value=99.99  "
        for((ad=9; ad < 10; ad=ad+1))
        do
          for((upd=0;upd<1;upd=upd+1))
          do
	        for((ssp=1536; ssp<=1536; ssp=256+ssp))
            do 
            	profiling/test_pd2.sh $1 "-quantization-calibration=${qc} -quantization-schema-constants=$qs -quantization-schema-activations=$as -aic-enable-depth-first -vtcm-working-set-limit-ratio=1.0 -size-split-granularity=${ssp} -allocator-dealloc-delay=$ad -use-producer-dma=$upd"
            done
          done
        done
      done
  done
done
