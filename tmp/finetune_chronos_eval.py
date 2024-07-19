
import numpy as np
# import pandas as pd
import torch
from gluonts.dataset.repository import get_dataset
from gluonts.dataset.split import split
from gluonts.ev.metrics import MASE, MeanWeightedSumQuantileLoss
from gluonts.itertools import batcher
from gluonts.model.evaluation import evaluate_forecasts
from gluonts.model.forecast import SampleForecast
from tqdm.auto import tqdm

from chronos import ChronosPipeline
# from gluonts.dataset.common import ListDataset
# import json
# import gzip


# datasets_lst = ['m4_hourly']
all_datasets = ['monash_australian_electricity_demand', 'monash_car_parts', 'monash_cif_2016', 'monash_covid_deaths', 'dominick', 'ercot', 'ett_small_15min', 'ett_small_1h', 'exchange_rate', 
                'monash_fred_md', 'monash_hospital', 'monash_m1_monthly', 'monash_m1_quarterly', 'monash_m1_yearly', 'monash_m3_monthly', 'monash_m3_quarterly', 'monash_m3_yearly', 'm4_quarterly', 'm4_yearly', 'm5', 
                'nn5_daily_without_missing', 'monash_nn5_weekly', 'monash_tourism_monthly', 'monash_tourism_quarterly', 'monash_tourism_yearly', 'monash_traffic', 'monash_weather']

available_datasets = ['covid_deaths', 'dominick', 'ercot', 'ett_small_15min', 'ett_small_1h', 'exchange_rate', 'm1_monthly', 'm1_quarterly', 'm1_yearly', 'm4_quarterly', 'm4_yearly', 
                'nn5_daily_without_missing', 'nn5_weekly', 'tourism_monthly', 'tourism_quarterly', 'tourism_yearly', 'traffic', 'weather']
missing_datasets = ['australian_electricity_demand', 'car_parts_without_missing', 'cif_2016', 'fred_md', 'hospital', 'm3_monthly', 'm3_quarterly', 'm3_yearly', 'm5']

local_datasets = ['australian_electricity_demand', 'car_parts_without_missing', 'monash_cif_2016', 'monash_covid_deaths', 'dominick', 'ercot', 'ett_small_15min', 'ett_small_1h', 'exchange_rate', 
                'monash_fred_md', 'monash_hospital', 'monash_m1_monthly', 'monash_m1_quarterly', 'monash_m1_yearly', 'monash_m3_monthly', 'monash_m3_quarterly', 'monash_m3_yearly', 'm4_quarterly', 'm4_yearly', 'm5', 
                'nn5_daily_without_missing', 'monash_nn5_weekly', 'monash_tourism_monthly', 'monash_tourism_quarterly', 'monash_tourism_yearly', 'monash_traffic', 'monash_weather']

# for name in available_datasets:
#     try:
#         # dataset_obj = get_dataset(name)
#         dataset_path = f'/root/.gluonts/datasets/{name}/train/data.json.gz'
#         with gzip.open(dataset_path, 'r') as f:
#             dataset = ListDataset([json.loads(line) for line in f], freq='H')#name_to_freq[name])
#         # print(f'{name}: {len(dataset)} samples')
#         print(f'DOWNLOADED DATASET {name}')
#         # entries = []
#         # for entry in dataset_obj.train:
#         #     entries.append(entry)
#         # df = pd.DataFrame(entries)
#         # print(df)
#     except:
#         print(f'Cannot get dataset {name}')


batch_size = 32
num_samples = 20
dataset = get_dataset("covid_deaths")
prediction_length = dataset.metadata.prediction_length

pipeline = ChronosPipeline.from_pretrained(
    "./tmp/output/chronos-covid_deaths/huggingface/ba1000/",  # fine-tuned on 'covid_deaths' for 1000 steps
    device_map="cuda:0",
    torch_dtype=torch.bfloat16,
)

# Split dataset for evaluation
_, test_template = split(dataset.test, offset=-prediction_length)
test_data = test_template.generate_instances(prediction_length)

# Generate forecast samples
forecast_samples = []
for batch in tqdm(batcher(test_data.input, batch_size=32)):
    context = [torch.tensor(entry["target"]) for entry in batch]
    forecast_samples.append(
        pipeline.predict(
            context,
            prediction_length=prediction_length,
            num_samples=num_samples,
        ).numpy()
    )
forecast_samples = np.concatenate(forecast_samples)

# Convert forecast samples into gluonts SampleForecast objects
sample_forecasts = []
for item, ts in zip(forecast_samples, test_data.input):
    forecast_start_date = ts["start"] + len(ts["target"])
    sample_forecasts.append(
        SampleForecast(samples=item, start_date=forecast_start_date)
    )

# Evaluate
metrics_df = evaluate_forecasts(
    sample_forecasts,
    test_data=test_data,
    metrics=[
        MASE(),
        MeanWeightedSumQuantileLoss(np.arange(0.1, 1.0, 0.1)),
    ],
)
print(metrics_df)

