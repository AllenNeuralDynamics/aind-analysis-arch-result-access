import json
import logging

import boto3
from aind_data_access_api.document_db import MetadataDbClient
from botocore.exceptions import ClientError
from dateutil import tz

DOC_DB_API_HOST = "api.allenneuraldynamics.org"

# for all analysis projects
ANALYSIS_DB_NAME = "analysis"
# for dynamic-foraging-analysis project
PROJECT = "dynamic-foraging-analysis"  # used for bucket name and docdb collection name
BUCKET = f"aind-{PROJECT}-prod-o5171v"


logging.basicConfig(level=logging.INFO)


def test_read_write_delete_docdb(session=None):
    record = {"_id": f"test_write_to_docdb"}

    docdb_api_client = MetadataDbClient(
        host=DOC_DB_API_HOST,
        database=ANALYSIS_DB_NAME,
        collection=PROJECT,
        boto_session=session,
    )
    resp_read = docdb_api_client._count_records()
    resp_write = docdb_api_client.upsert_one_docdb_record(record)
    resp_delete = docdb_api_client.delete_one_record(record["_id"])

    logging.info(
        f"DocDB: read ({resp_read})/ write ({resp_write.status_code})/ delete ({resp_delete.status_code})"
    )
    return


def test_read_write_delete_s3(session=None):
    key = "test/test_write_to_S3.json"
    file = {"_id": "test_write_to_S3"}
    file_json = json.dumps(file, indent=3, sort_keys=True)

    s3_client = session.client("s3") if session is not None else boto3.client("s3")
    resp_read = s3_client.list_objects_v2(Bucket=BUCKET)["ResponseMetadata"]["HTTPStatusCode"]
    try:
        resp_write = s3_client.put_object(Bucket=BUCKET, Key=key, Body=file_json)[
            "ResponseMetadata"
        ]["HTTPStatusCode"]
    except ClientError as e:
        resp_write = e.response["Error"]["Code"]
    try:
        resp_delete = s3_client.delete_object(Bucket=BUCKET, Key="test/test_write_to_S3.json")[
            "ResponseMetadata"
        ]["HTTPStatusCode"]
    except ClientError as e:
        resp_delete = e.response["Error"]["Code"]

    logging.info(f"S3: read ({resp_read})/ write ({resp_write})/ delete ({resp_delete})")
    return


if __name__ == "__main__":
    # read/write/delete should all work if capsule has attached aind-codeocean-user role
    # do not need to create session explicitly
    test_read_write_delete_docdb()
    test_read_write_delete_s3()
