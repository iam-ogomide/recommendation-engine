import os
import sys
from datetime import datetime, timezone
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
db = MongoClient(os.environ["MONGO_URI"])[os.environ["MONGO_DB"]]
coll = db["runner_bureau_snapshots"]

if len(sys.argv) > 2 and sys.argv[1] == "bump":
    # make the EARLIER snapshot's score higher by N points, to simulate a drop
    res = coll.update_one(
        {"subscriber_id": "SUB-0001", "snapshot_date": {"$lt": datetime(2026, 10, 22, tzinfo=timezone.utc)}},
        {"$inc": {"score": int(sys.argv[2])}},
    )
    print("bumped:", res.modified_count)

for d in coll.find({"subscriber_id": "SUB-0001"}, {"_id": 0}).sort("snapshot_date", 1):
    print(d["snapshot_date"].date(), d["bureau"], "score:", d.get("score"),
          "npl:", d.get("non_performing_count"), d.get("flags"))