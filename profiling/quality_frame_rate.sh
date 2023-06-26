#!/usr/bin/env bash
for qc in  KLMinimization # KLMinimizationV2 MSE SQNR None  Percentile
do
  for qs in symmetric # symmetric_with_uint8 asymmetric # symmetric_with_power2_scale
  do
    for as in asymmetric # symmetric_with_uint8  symmetric #symmetric_with_power2_scale
    do
      [ "$qc" == "Percentile" ] && qc="${qc} -percentile-calibration-value=99.99  "
      for((ad=4; ad<=9; ad=ad+1))
      do
        for((upd=1;upd<=1;upd=upd+1))
        do
          #for((ssp=522; ssp<=2048; ssp=36+ssp))
          for((ssp=1536; ssp<=2048; ssp=128+ssp))
          do
            for((r=10; r<=10; r=r+1))
            do
              ratio=$(echo "scale=1; ${r}/10.0" | bc | sed -e 's/^\./0\./g')
              for((o=1; o<=1; o=o+1))
              do
                set -x
                profiling/test_frame_rate.sh $1 "-ols=$o -quantization-calibration=${qc} -quantization-schema-constants=$qs -quantization-schema-activations=$as -aic-enable-depth-first -vtcm-working-set-limit-ratio=${ratio} -size-split-granularity=${ssp} -allocator-dealloc-delay=$ad -use-producer-dma=$upd"
                set +x
              done
            done
          done
        done
      done
    done
  done
done
