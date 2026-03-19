import asyncio
from store import MATCHES
from judge import judge_internal

QUEUE = asyncio.Queue()

async def judge_worker():

    while True:

        match_id = await QUEUE.get()

        match = MATCHES.get(match_id)

        if not match:
            continue

        result = judge_internal(
            match["referenceUrl"],
            match["artA"],
            match["artB"]
        )

        match["result"] = result
        match["state"] = "FINISHED"
