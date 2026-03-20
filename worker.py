import asyncio
from db import get_conn
from judge import judge_internal

QUEUE = asyncio.Queue()

async def judge_worker():
    print("Worker started")

    while True:
        match_id = await QUEUE.get()

        try:
            conn = get_conn()
            cur = conn.cursor()

            print("Processing:", match_id)

            # ===============================
            # GET MATCH
            # ===============================
            cur.execute(
                "SELECT reference_index, art_a, art_b, state FROM matches WHERE match_id=%s",
                (match_id,)
            )
            row = cur.fetchone()

            if not row:
                print("Match missing")
                continue

            ref_index, artA, artB, state = row

            if state != "JUDGING":
                print("Not in judging state, skipping")
                continue

            if not artA or not artB:
                print("Missing art, skipping")
                continue

            # ===============================
            # JUDGE
            # ===============================
            try:
                result = judge_internal(ref_index, artA, artB)
            except Exception as e:
                print("Judge crash:", e)
                continue

            if not result:
                print("Result is null, skipping")
                continue

            # ===============================
            # SAVE RESULT
            # ===============================
            cur.execute("""
                UPDATE matches
                SET state='FINISHED', result=%s
                WHERE match_id=%s AND state='JUDGING'
            """, (str(result), match_id))

            conn.commit()

            print("Match finished:", match_id)

        except Exception as e:
            print("Worker error:", e)

        await asyncio.sleep(0.1)
