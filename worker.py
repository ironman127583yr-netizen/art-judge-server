import asyncio
from judge import judge_internal

MATCHES = {}
QUEUE = asyncio.Queue()

async def judge_worker():
    print("Judge worker started")

    while True:
        match_id = await QUEUE.get()

        match = MATCHES.get(match_id)
        if not match:
            continue

        try:
            result = judge_internal(
                match["referenceUrl"],
                match["artA"],
                match["artB"]
            )

            match["result"] = result
            match["state"] = "FINISHED"

            print(f"Match {match_id} finished")

        except Exception as e:
            print("Worker error:", e)
