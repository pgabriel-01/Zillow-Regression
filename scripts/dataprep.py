from azureml.core import Run

import pandas as pd 
import numpy as np 
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from azureml.core import Run
import mlflow
import pickle
import os

mlflow.autolog()

# Constants
RANDOM_SEED=42

# Get the parameters
parser = argparse.ArgumentParser()
parser.add_argument('--output_path', dest='output_path', required=True)
args = parser.parse_args()
zillow_housing_dataset = Run.get_context().input_datasets['house_prices_train']

# Get the experiment/job run context
run = Run.get_context()

# Load the data (passed as an input dataset)
df_train = zillow_housing_dataset.to_pandas_dataframe()

# Save the prepped data
print('Saving Data...')
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
pq.write_table(pa.Table.from_pandas(df_train), args.output_path)

# End the run
print(f"Wrote test to {args.output_path} and train to {args.output_path}")
run.complete()
