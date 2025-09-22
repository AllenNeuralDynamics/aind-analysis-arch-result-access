import pandas as pd
import os
import logging

from codeocean import CodeOcean
from codeocean.data_asset import DataAssetAttachParams

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

if __name__ == "__main__":
    
    logging.basicConfig(level=logging.INFO)

    asset_ids = get_asset_ids()
    mounted = attach_assets(asset_ids)