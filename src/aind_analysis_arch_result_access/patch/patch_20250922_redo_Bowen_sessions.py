import pandas as pd
import os
import logging

from codeocean import CodeOcean
from codeocean.data_asset import DataAssetAttachParams
import shutil

logger = logging.getLogger(__name__)

client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("CO_API"))

def get_asset_ids(csv_file = '~/capsule/data/Bowen_IncompleteSessions-081225.csv'):
    df = pd.read_csv(csv_file)
    return df.iloc[:, 0].tolist()

def attach_assets(asset_ids):
    mounted = []
    logger.info(f"Attaching {len(asset_ids)} assets")
    for asset_id in asset_ids:
        try:
            logger.info(f"Attaching asset {asset_id}")
            data_asset = DataAssetAttachParams(
                    id=asset_id,
                )
            results = client.capsules.attach_data_assets(
                    capsule_id=os.getenv("CO_CAPSULE_ID"),
                    attach_params=[data_asset],
                )
            mounted.append({"id": asset_id, "mount": client.data_assets.get_data_asset(asset_id).mount})
        except Exception as e:
            logger.error(f"Failed to attach asset {asset_id}: {e}")
    logger.info(f"Attached {len(mounted)} assets successfully")
    return mounted

def extract_all_nwbs(mounted):
    # Copy the nwb file from mounted_directory/nwb to /results/extracted_Bowen_nwbs
    output_dir = '/results/extracted_Bowen_nwbs'
    os.makedirs(output_dir, exist_ok=True)
    for asset in mounted:
        try:
            nwb_source = os.path.join('/root/capsule/data', asset['mount'], 'nwb')
            if os.path.exists(nwb_source):
                for nwb_dir in os.listdir(nwb_source):
                    dir_path = os.path.join(nwb_source, nwb_dir)
                    if os.path.isdir(dir_path):
                        dst_dir = os.path.join(output_dir, nwb_dir)
                        logger.info(f"Copying directory {dir_path} to {dst_dir}")
                        if os.path.exists(dst_dir):
                            shutil.rmtree(dst_dir)
                        shutil.copytree(dir_path, dst_dir)
            else:
                logger.warning(f"No 'nwb' directory found in asset {asset['id']} at {nwb_source}")
        except Exception as e:
            logger.error(f"Failed to extract NWB files from asset {asset['id']}: {e}")
            

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)

    asset_ids = get_asset_ids()
    mounted = attach_assets(asset_ids)
    
    extract_all_nwbs(mounted)
    