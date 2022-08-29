from azureml.core import Run

import pandas as pd 
import numpy as np 
import pyarrow as pa
import pyarrow.parquet as pq
import argparse
from azureml.core import Run
import mlflow
import pickle

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

# drop the NaN values of the dataset that relates to classification
# def prepare_data(dataframe):
#     from azureml.training.tabular.preprocessing import data_cleaning
    
#     label_column_name = 'SalePrice'
    
#     # extract the features, target and sample weight arrays
#     y = dataframe[label_column_name].values
#     X = dataframe.drop([label_column_name], axis=1)
#     sample_weights = None
#     X, y, sample_weights = data_cleaning._remove_nan_rows_in_X_y(X, y, sample_weights,
#      is_timeseries=False, target_column=label_column_name)
    
#     return X, y, sample_weights

def prepare_train_x(df):
    train_x = df.dropna(inplace=True)
    return train_x

prepare_train_x_df=prepare_train_x(df_train)

# Save the prepped data
print('Saving Data...')
os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
pq.write_table(pa.Table.from_pandas(prepare_train_x_df), args.output_path)

# End the run
print(f"Wrote test to {args.output_path} and train to {args.output_path}")
run.complete()
