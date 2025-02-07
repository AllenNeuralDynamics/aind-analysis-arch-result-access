'''
Util functions for public S3 bucket access
'''
import s3fs
import pickle

# The processed bucket is public
fs = s3fs.S3FileSystem(anon=True)


def get_df_from_s3_pkl(s3_path):
    '''
    Load a pickled dataframe from an s3 path
    '''
    with fs.open(s3_path, 'rb') as f:
        df = pickle.load(f)
    return df
