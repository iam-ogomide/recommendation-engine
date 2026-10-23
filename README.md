# FICO Snapshot

A small command-line project that finds a MongoDB document by its `bvn` field,
saves its `fico` data and snapshot metadata as JSON, and inserts the same data
into the `fico_consumer_data` collection in the configured database.

The sample document in `sample-fico-Completedata.txt` uses the expected shape:
`bvn`, `name`, and nested `fico` fields.

## Setup

Requires Python 3.10 or newer and access to the MongoDB database and collection.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
py -m pip install -e ".[test]"
```

Create a `.env` file in the project root (the same folder as `pyproject.toml`)
by copying `.env.example`, then fill in your MongoDB connection details:

```powershell
Copy-Item .env.example .env
```

The CLI loads this root-level `.env` automatically. `.env` is ignored by Git so
your actual connection details are not committed.

For a hosted database, use the connection URI supplied for that MongoDB
deployment. Keep credentials in environment variables or a secret manager; do
not place them in source control.

## Create a snapshot

```powershell
fico-snapshot 22231186083
```

Or run the module directly:

```powershell
py -m fico_snapshot.cli 22231186083
```

You can also run the root-level launcher:

```powershell
py fico_snapshot.py 22231186083
```

The command prints the created file path. Files are written to
`snapshots/{sanitized-name}-{generated-id}/{YYYY-MM-DD}.json`; each invocation gets a unique directory. The JSON and database document contain
`fico`, `firstCentral`, `bvn`, `name`, `snapshot_date` (a UTC BSON Date at
midnight on that date), and `pulled_at` (UTC timestamp). Since JSON has no
native date type, `snapshot_date` appears in Extended JSON as
`{"$date":"YYYY-MM-DDT00:00:00Z"}`.
The database collection is created automatically by MongoDB on the first
successful insert. BSON-specific values in JSON, including `pulled_at`, are
encoded using MongoDB Extended JSON.

## Tests

```powershell
py -m pytest
```
