"""Init package"""

__version__ = "0.8.0"

from aind_data_access_api.document_db import MetadataDbClient

S3_PATH_BONSAI_ROOT = "s3://aind-behavior-data/foraging_nwb_bonsai_processed"
S3_PATH_BPOD_ROOT = "s3://aind-behavior-data/foraging_nwb_bpod_processed"
S3_PATH_ANALYSIS_ROOT = "s3://aind-dynamic-foraging-analysis-prod-o5171v"

# New collection
analysis_docDB_dft = MetadataDbClient(
    host="api.allenneuraldynamics.org",
    database="analysis",
    collection="dynamic-foraging-analysis",
)
