# DeepDB: Learn from Data, not from Queries!

DeepDB is a data-driven learned database component achieving state-of-the-art-performance in cardinality estimation and 
approximate query processing (AQP). This is the implementation described in 

Benjamin Hilprecht, Andreas Schmidt, Moritz Kulessa, Alejandro Molina, Kristian Kersting, Carsten Binnig: 
"DeepDB: Learn from Data, not from Queries!", VLDB'2020. [[PDF]](https://arxiv.org/abs/1909.00607)

![DeepDB Overview](baselines/plots/overview.png "DeepDB Overview")

# Setup
```
git clone https://github.com/DataManagementLab/deepdb-public.git
cd deepdb-public
sudo apt install -y libpq-dev gcc python3-dev
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

# Reproduce Experiments

## Cardinality Estimation
Download the [Job dataset](http://homepages.cwi.nl/~boncz/job/imdb.tgz).
Generate hdf files from csvs.
```
python3 maqp.py --generate_hdf
    --dataset imdb-light
    --csv_seperator ,
    --csv_path ../imdb-benchmark
    --hdf_path ../imdb-benchmark/gen_single_light
    --max_rows_per_hdf_file 100000000
```

Generate sampled hdf files from csvs.
```
python3 maqp.py --generate_sampled_hdfs
    --dataset imdb-light
    --hdf_path ../imdb-benchmark/gen_single_light
    --max_rows_per_hdf_file 100000000
    --hdf_sample_size 10000
```

Learn ensemble with the optimized rdc strategy (requires postgres with imdb dataset)
```
python3 maqp.py --generate_ensemble
    --dataset imdb-light 
    --samples_per_spn 10000000 10000000 1000000 1000000 1000000
    --ensemble_strategy rdc_based
    --hdf_path ../imdb-benchmark/gen_single_light
    --max_rows_per_hdf_file 100000000
    --samples_rdc_ensemble_tests 10000
    --ensemble_path ../imdb-benchmark/spn_ensembles
    --database_name imdb
    --post_sampling_factor 10 10 5 1 1
    --ensemble_budget_factor 5
    --ensemble_max_no_joins 3
    --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl
```

Alternatively: Learn base ensemble over different tables with naive strategy. 
(Does not work with different dataset sizes because join sizes are hard coded but does not require postgres)
```
python3 maqp.py --generate_ensemble
    --dataset imdb-light 
    --samples_per_spn 1000000 1000000 1000000 1000000 1000000
    --ensemble_strategy relationship
    --hdf_path ../imdb-benchmark/gen_single_light
    --ensemble_path ../imdb-benchmark/spn_ensembles
    --max_rows_per_hdf_file 100000000
    --post_sampling_factor 10 10 5 1 1
```

Evaluate performance for queries.
```
python3 maqp.py --evaluate_cardinalities
    --rdc_spn_selection
    --max_variants 1
    --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl
    --dataset imdb-light
    --target_path ./baselines/cardinality_estimation/results/deepDB/imdb_light_model_based_budget_5.csv
    --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl
    --query_file_location ./benchmarks/job-light/sql/job_light_queries.sql
    --ground_truth_file_location ./benchmarks/job-light/sql/job_light_true_cardinalities.csv
```

## Updates

Conditional incremental learning (i.e., initial learning of all films before 2013, newer films learn incremental)
```
python3 maqp.py  --generate_ensemble
    --dataset imdb-light
    --samples_per_spn 10000000 10000000 1000000 1000000 1000000
    --ensemble_strategy rdc_based
    --hdf_path ../imdb-benchmark/gen_single_light
    --max_rows_per_hdf_file 100000000
    --samples_rdc_ensemble_tests 10000
    --ensemble_path ../imdb-benchmark/spn_ensembles
    --database_name JOB-light
    --post_sampling_factor 10 10 5 1 1
    --ensemble_budget_factor 0
    --ensemble_max_no_joins 3
    --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl
    --incremental_condition "title.production_year<2013"
```

## Optimized Inference
Generate the C++ code. (Currently only works for cardinality estimation).
```
python3 maqp.py --code_generation 
    --ensemble_path ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl
```

Compile it in a venv with pybind installed. 
Sometimes installing this yields: `ModuleNotFoundError: No module named 'pip.req'`
One workaround is to downgrade pip `pip3 install pip==9.0.3` as described [here](https://stackoverflow.com/questions/25192794/no-module-named-pip-req).

The command below works for ubuntu 18.04. Make sure the generated .so file is in the root directory of the project.
```
g++ -O3 -Wall -shared -std=c++11 -ftemplate-depth=2048 -ftime-report -fPIC `python3 -m pybind11 --includes` optimized_inference.cpp -o optimized_inference`python3-config --extension-suffix`
```

If you now want to leverage the module you have to specify it for cardinalities.
```
python3 maqp.py --evaluate_cardinalities 
    --rdc_spn_selection 
    --max_variants 1 
    --pairwise_rdc_path ../imdb-benchmark/spn_ensembles/pairwise_rdc.pkl 
    --dataset imdb-light 
    --target_path ./baselines/cardinality_estimation/results/deepDB/imdb_light_model_based_budget_5.csv 
    --ensemble_location ../imdb-benchmark/spn_ensembles/ensemble_join_3_budget_5_10000000.pkl 
    --query_file_location ./benchmarks/job-light/sql/job_light_queries.sql 
    --ground_truth_file_location ./benchmarks/job-light/sql/job_light_true_cardinalities.csv 
    --use_generated_code
```

## AQP
### SSB pipeline

Generate standard SSB dataset (Scale Factor=500) and use the correct seperator.
```
for i in `ls *.tbl`; do
    sed 's/|$//' $i > $TMP_DIR/${i/tbl/csv} &
    echo $i;
done
```
Create lineorder sample
```
cat lineorder.csv | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .003333) print $0}' > lineorder_sampled.csv
```

Generate hdf files from csvs.
```
python3 maqp.py --generate_hdf
    --dataset ssb-500gb
    --csv_seperator \|
    --csv_path ../mqp-data/ssb-benchmark
    --hdf_path ../mqp-data/ssb-benchmark/gen_hdf
```

Learn the ensemble with a naive strategy.
```
python3 maqp.py --generate_ensemble 
    --dataset ssb-500gb
    --samples_per_spn 1000000
    --ensemble_strategy single 
    --hdf_path ../mqp-data/ssb-benchmark/gen_hdf 
    --ensemble_path ../mqp-data/ssb-benchmark/spn_ensembles
    --rdc_threshold 0.3
    --post_sampling_factor 10
```

Optional: Compute ground truth for AQP queries (requires postgres with ssb schema).
```
python3 maqp.py --aqp_ground_truth
    --query_file_location ./benchmarks/ssb/sql/aqp_queries.sql
    --target_path ./benchmarks/ssb/ground_truth_500GB.pkl
    --database_name ssb
```

Evaluate the AQP queries.
```
python3 maqp.py --evaluate_aqp_queries
    --dataset ssb-500gb
    --target_path ./baselines/aqp/results/deepDB/ssb_500gb_model_based.csv
    --ensemble_location ../mqp-data/ssb-benchmark/spn_ensembles/ensemble_single_ssb-500gb_1000000.pkl
    --query_file_location ./benchmarks/ssb/sql/aqp_queries.sql
    --ground_truth_file_location ./benchmarks/ssb/ground_truth_500GB.pkl
```

Optional: Create the ground truth for confidence interval. (with 10M because we also use 10M samples for the training)
```
python3 maqp.py --aqp_ground_truth
    --query_file_location ./benchmarks/ssb/sql/confidence_queries.sql
    --target_path ./benchmarks/ssb/confidence_intervals/confidence_interval_10M.pkl
    --database_name ssb
```

Evaluate the confidence intervals.
```
python3 maqp.py --evaluate_confidence_intervals
    --dataset ssb-500gb
    --target_path ./baselines/aqp/results/deepDB/ssb500GB_confidence_intervals.csv
    --ensemble_location ../mqp-data/ssb-benchmark/spn_ensembles/ensemble_single_ssb-500gb_1000000.pkl
    --query_file_location ./benchmarks/ssb/sql/aqp_queries.sql
    --ground_truth_file_location ./benchmarks/ssb/confidence_intervals/confidence_interval_10M.pkl
    --confidence_upsampling_factor 300
    --confidence_sample_size 10000000
```

### Flights pipeline
Generate flights dataset with scale factor 1 billion using [IDEBench](https://github.com/IDEBench/IDEBench-public) and generate a sample using
```
cat dataset.csv | awk 'BEGIN {srand()} !/^$/ { if (rand() <= .01) print $0}' > dataset_sampled.csv
```

Generate hdf files from csvs.
```
python3 maqp.py --generate_hdf
    --dataset flights1B
    --csv_seperator ,
    --csv_path ../mqp-data/flights-benchmark
    --hdf_path ../mqp-data/flights-benchmark/gen_hdf
```

Learn the ensemble.
```
python3 maqp.py --generate_ensemble 
    --dataset flights1B
    --samples_per_spn 10000000 
    --ensemble_strategy single 
    --hdf_path ../mqp-data/flights-benchmark/gen_hdf 
    --ensemble_path ../mqp-data/flights-benchmark/spn_ensembles
    --rdc_threshold 0.3
    --post_sampling_factor 10
```

Optional: Compute ground truth
```
python3 maqp.py --aqp_ground_truth
    --dataset flights1B
    --query_file_location ./benchmarks/flights/sql/aqp_queries.sql
    --target_path ./benchmarks/flights/ground_truth_1B.pkl
    --database_name flights   
```

Evaluate the AQP queries.
```  
python3 maqp.py --evaluate_aqp_queries
    --dataset flights1B
    --target_path ./baselines/aqp/results/deepDB/flights1B_model_based.csv
    --ensemble_location ../mqp-data/flights-benchmark/spn_ensembles/ensemble_single_flights1B_10000000.pkl
    --query_file_location ./benchmarks/flights/sql/aqp_queries.sql
    --ground_truth_file_location ./benchmarks/flights/ground_truth_1B.pkl
```

Optional: Create the ground truth for confidence interval. (with 10M because we also use 10M samples for the training)
```
python3 maqp.py --aqp_ground_truth
    --dataset flights1B
    --query_file_location ./benchmarks/flights/sql/confidence_queries.sql
    --target_path ./benchmarks/flights/confidence_intervals/confidence_interval_10M.pkl
    --database_name flights10M_origsample 
```

Evaluate the confidence intervals.
```
python3 maqp.py --evaluate_confidence_intervals
    --dataset flights1B
    --target_path ./baselines/aqp/results/deepDB/flights1B_confidence_intervals.csv
    --ensemble_location ../mqp-data/flights-benchmark/spn_ensembles/ensemble_single_flights1B_10000000.pkl
    --query_file_location ./benchmarks/flights/sql/aqp_queries.sql
    --ground_truth_file_location ./benchmarks/flights/confidence_intervals/confidence_interval_10M.pkl
    --confidence_upsampling_factor 100
    --confidence_sample_size 10000000
```
