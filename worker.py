import asyncio
from store import MATCHES
from judge import judge_internal

QUEUE = asyncio.Queue()

async def judge_worker():
    print("Judge worker started")

    while True:
        match_id = await QUEUE.get()

        try:
            match = MATCHES.get(match_id)

            if not match:
                continue

            print("Processing match:", match_id)

            result = judge_internal(
                match["referenceUrl"],
                match["artA"],
                match["artB"]
            )

            match["result"] = result
            match["state"] = "FINISHED"

        except Exception as e:
            print("Worker error:", e)
