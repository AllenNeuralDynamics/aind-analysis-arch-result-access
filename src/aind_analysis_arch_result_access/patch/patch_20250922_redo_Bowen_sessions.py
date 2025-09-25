"""
Fix Bowen' sessions. See issue
https://github.com/AllenNeuralDynamics/aind-analysis-arch-result-access/issues/26
"""

import logging
import os
import shutil

import pandas as pd
from aind_data_access_api.document_db import MetadataDbClient
from codeocean import CodeOcean
from codeocean.data_asset import DataAssetAttachParams

logger = logging.getLogger(__name__)

client = CodeOcean(domain="https://codeocean.allenneuraldynamics.org", token=os.getenv("CO_API"))

prod_db_client = MetadataDbClient(
    host="api.allenneuraldynamics.org",
    database="analysis",
    collection="dynamic-foraging-analysis",
)


def get_asset_ids(csv_file="~/capsule/data/Bowen_IncompleteSessions-081225.csv"):
    """Get asset ids to fix
    """
    df = pd.read_csv(csv_file)
    return df.iloc[:, 0].tolist()


def attach_assets(asset_ids):
    """Attach to current capsule"""
    mounted = []
    logger.info(f"Attaching {len(asset_ids)} assets")
    for asset_id in asset_ids:
        try:
            logger.info(f"Attaching asset {asset_id}")
            data_asset = DataAssetAttachParams(
                id=asset_id,
            )
            client.capsules.attach_data_assets(
                capsule_id=os.getenv("CO_CAPSULE_ID"),
                attach_params=[data_asset],
            )
            mounted.append(
                {"id": asset_id, "mount": client.data_assets.get_data_asset(asset_id).mount}
            )
        except Exception as e:
            logger.error(f"Failed to attach asset {asset_id}: {e}")
    logger.info(f"Attached {len(mounted)} assets successfully")
    return mounted


def extract_all_nwbs(mounted):
    """Extract all nwb files from mounted assets to /results/extracted_Bowen_nwbs"""
    output_dir = "/results/extracted_Bowen_nwbs"
    os.makedirs(output_dir, exist_ok=True)

    import concurrent.futures

    def process_asset(asset):
        try:
            nwb_source = os.path.join("/root/capsule/data", asset["mount"], "nwb")
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

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        executor.map(process_asset, mounted)


def remove_records_from_docDB(mounted):
    """Remove records from docDB for the given mounted assets"""

    for this in mounted:
        # Get docDB query
        name = this["mount"]
        splits = name.split("_")
        subject_id = splits[1]
        session_date = splits[2]

        record_filter = {"subject_id": subject_id, "session_date": session_date}

        # Delete records from docDB
        before_count = len(
            prod_db_client._get_records(filter_query=record_filter, projection={"_id": 1})
        )
        results = prod_db_client._delete_many_records(
            record_filter={"subject_id": subject_id, "session_date": session_date}
        )
        after_count = len(
            prod_db_client._get_records(filter_query=record_filter, projection={"_id": 1})
        )
        if results.status_code == 200:
            logger.info(
                f"Deleted {before_count - after_count} records for subject "
                f"{subject_id} on date {session_date}"
            )
        else:
            logger.error(
                f"Failed to delete records for subject {subject_id} on "
                f" date {session_date}: {results.raw_response}"
            )


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    asset_ids = get_asset_ids()

    mounted = attach_assets(asset_ids)

    # extract_all_nwbs(mounted)

    # --- THIS CANNOT BE UNDONE!! ---
    # remove_records_from_docDB(mounted)
