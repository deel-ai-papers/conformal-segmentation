python experiments/crc_test_metrics.py \
     --gpu ${2} \
     --input experiments/outputs/processed_lambdas/crc_${1}_LoveDA_optimized_lambdas.csv 
python experiments/crc_test_metrics.py \
     --gpu ${2} \
     --input experiments/outputs/processed_lambdas/crc_${1}_LoveDA_optimized_lambdas_cov_0.9.csv 