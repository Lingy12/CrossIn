python build_dataset_with_alignment.py data/alpaca data/alpaca-align-all "en,zh,vi,es" -1
python build_dataset_with_alignment.py data/alpaca data/alpaca-align-all-3000 "en,zh,vi,es" 3000
python build_dataset_with_alignment.py data/alpaca data/alpaca-align-all-6000 "en,zh,vi,es" 6000
python build_dataset_with_alignment.py data/alpaca data/alpaca-align-all-15000 "en,zh,vi,es" 15000

python build_dataset.py data/platypus data/platypus-unified-en "en," -1

python combine_ds.py --ds_lst "['data/platypus-unified-en','data/alpaca-align-all']"  --output_name data/platypus-alpaca-comb
python combine_ds.py --ds_lst "['data/platypus-unified-en','data/alpaca-align-all-3000']"  --output_name data/platypus-alpaca-comb-3000
python combine_ds.py --ds_lst "['data/platypus-unified-en','data/alpaca-align-all-6000']"  --output_name data/platypus-alpaca-comb-6000
python combine_ds.py --ds_lst "['data/platypus-unified-en','data/alpaca-align-all-15000']"  --output_name data/platypus-alpaca-comb-15000
