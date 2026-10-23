# Subscriber data uploader

Uploads subscriber records from a CSV or Excel workbook to MongoDB, using the
`subscriber_id` column as the record key. Existing matching records are replaced
with the current row, so rerunning the upload does not add duplicate subscribers.

## Set up

From this project folder, create your local environment file by copying the
example:

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:

- `MONGODB_URI` — your MongoDB connection URI
- `MONGODB_DATABASE` — the database name you provide
- `MONGODB_COLLECTION` — leave as `demo_subscribers_data` unless you want a different collection
- `SNAPSHOT_COLLECTION` — collection for BVN lookup snapshots; defaults to `demo_subscriber_snapshots`
- `DATA_FILE` — defaults to `subscribers_sim.csv`

Keep `.env` private; it contains database credentials and is excluded from Git.
If your workbook contains BVNs, store that column as text in Excel to preserve
leading zeros.

Install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run

Upload the file configured by `DATA_FILE`:

```powershell
python upload_to_mongodb.py
```

Or pass a specific CSV or `.xlsx`/`.xlsm` file:

```powershell
python upload_to_mongodb.py .\subscribers_sim.csv
python upload_to_mongodb.py .\subscribers.xlsx
```

The input file must have a header row containing a unique `subscriber_id`
column. Date columns named `subscribed_at`, `last_payment_date`, and
`next_run_date` are stored as MongoDB dates. `cycle_day` and
`completed_cycles` are stored as integers. Other values, including BVNs, are
preserved as text where the file format supplies them as text.

## Create a subscriber snapshot

Look up a subscriber by BVN from `MONGODB_COLLECTION` and save the snapshot
both as a JSON file and as a document in `SNAPSHOT_COLLECTION`:

```powershell
python get_subscriber_snapshot.py 99601815908
```

The JSON file is written under
`snapshots/<name>-<generated-id>/<YYYY-MM-DD>.json`. The JSON and MongoDB
document contain `name`, `subscriber_id`, `bvn`, `plan`, `status`,
`cycle_date`, `snapshot_date`, and `pulled_at`. `cycle_date` is taken from the
subscriber's `next_run_date`. Snapshot dates and pull timestamps are UTC.
The local `snapshots/` folder is excluded from Git because its contents include
personal data.
