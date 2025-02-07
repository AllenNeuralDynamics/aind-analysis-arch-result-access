'''
Get results from Han's pipeline
https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline
'''

import s3fs
import pickle
import pandas

s3_path = 's3://aind-behavior-data/foraging_nwb_bonsai_processed/df_sessions.pkl'

# The processed bucket is public
fs = s3fs.S3FileSystem(anon=True)

# Open the file and load the pickle
with fs.open(s3_path, 'rb') as f:
    data = pickle.load(f)

print(data)