import asyncio
import json
from db import get_conn
from judge import judge_internal

QUEUE = asyncio.Queue()

REFERENCE_POOL = [
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/0.jpg",
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/1.jpg",
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/2.jpg",
    "https://raw.githubusercontent.com/ironman127583yr/ArtChess-reference/main/3.jpg",
]

async def judge_worker():
    print("Judge worker started")

    while True:
        match_id = await QUEUE.get()

        conn = get_conn()
        cur = conn.cursor()

        try:
            cur.execute("""
                SELECT reference_index, art_a, art_b
                FROM matches
                WHERE match_id=%s
            """, (match_id,))

            row = cur.fetchone()

            if not row:
                continue

            ref_index, artA, artB = row
            reference_url = REFERENCE_POOL[ref_index]

            result = judge_internal(reference_url, artA, artB)

            cur.execute("""
                UPDATE matches
                SET state='FINISHED', result=%s
                WHERE match_id=%s
            """, (json.dumps(result), match_id))

            conn.commit()

            print("Finished:", match_id)

        except Exception as e:
            print("Worker error:", e)
