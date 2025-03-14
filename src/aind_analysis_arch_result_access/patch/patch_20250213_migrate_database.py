"""Patch script to migrate records from old collection in dev to new collection in prod"""

from aind_data_access_api.document_db import MetadataDbClient

from aind_analysis_arch_result_access.util.reformat import split_nwb_name

# Old collection
dev_db_client = MetadataDbClient(
    host="api.allenneuraldynamics-test.org",
    database="analysis",
    collection="dynamic_foraging_analysis",
)

# New collection
prod_db_client = MetadataDbClient(
    host="api.allenneuraldynamics.org",
    database="analysis",
    collection="dynamic-foraging-analysis",
)


def modify_and_migrate_records_in_batch(skip, limit):
    """Migrate records from old collection to new collection in batches of size limit"""

    records_this_batch = dev_db_client._get_records(
        filter_query=None,
        projection={
            "analysis_results.fit_settings.fit_choice_history": 0,
            "analysis_results.fit_settings.fit_reward_history": 0,
        },
        limit=limit,
        skip=skip,
        sort=[("_id", 1)],
    )

    # Modify the records
    for record in records_this_batch:
        # Add subject_id and session_date for faster querying later
        subject_id, session_date, nwb_suffix = split_nwb_name(record["nwb_name"])
        record["subject_id"] = subject_id
        record["session_date"] = session_date

        # Overwrite s3 path to the new prod bucket
        if record["status"] == "success":
            record["s3_location"] = (
                f"s3://aind-dynamic-foraging-analysis-prod-o5171v/{record['_id']}"
            )
        else:
            record["s3_location"] = None

    # Upsert to new database
    response = prod_db_client.upsert_list_of_docdb_records(records_this_batch)

    return response, len(records_this_batch)


if __name__ == "__main__":
    # Migrate all records from old collection to new collection
    skip = 0
    limit = 5000
    while True:
        response, n_this_batch = modify_and_migrate_records_in_batch(skip, limit)
        print(f"Batch {skip//limit}: {set(response)} ({n_this_batch} records)")
        skip += limit
        if n_this_batch < limit:
            break
