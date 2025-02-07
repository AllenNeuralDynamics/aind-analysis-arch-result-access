'''
Get results from Han's pipeline
https://github.com/AllenNeuralDynamics/aind-foraging-behavior-bonsai-trigger-pipeline
'''

import logging
from aind_analysis_arch_result_access.util.s3 import get_df_from_s3_pkl

logger = logging.getLogger(__name__)

s3_path_root = 's3://aind-behavior-data/foraging_nwb_bonsai_processed'

def get_session_table():
    # Load the session table from s3
    logger.info('Loading session table from s3')
    df_session = get_df_from_s3_pkl(f'{s3_path_root}/df_sessions.pkl')
        
    return df_session


if __name__ == '__main__':
    df = get_session_table()
    print(df.head())