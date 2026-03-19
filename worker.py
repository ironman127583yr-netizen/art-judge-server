import asyncio
from store import MATCHES
from judge import judge_internal

QUEUE = asyncio.Queue()

async def judge_worker():

    print("Judge worker started")

    while True:

        try:
            jobs = get_object("judge_queue")

            if not jobs:
                await asyncio.sleep(2)
                continue

            if not isinstance(jobs, list):
                jobs = [jobs]

            for job in jobs:

                match_id = job["matchId"]

                print("Processing match", match_id)

                result = judge_internal(
                    job["referenceUrl"],
                    job["artAUrl"],
                    job["artBUrl"]
                )

                match = get_object("match_" + match_id)

                if match is None:
                    continue

                match["result"] = result
                match["state"] = "FINISHED"

                set_object("match_" + match_id, match)

            delete_object("judge_queue")

        except Exception as e:
            print("Worker error:", e)

        await asyncio.sleep(2)
