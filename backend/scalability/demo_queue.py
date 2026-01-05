import time
import threading
from queue import Queue

job_queue = Queue()

def worker(worker_id):
    while True:
        job = job_queue.get()
        if job is None:
            break
        print(f"[Worker {worker_id}] Processing job: {job}")
        time.sleep(2)
        print(f"[Worker {worker_id}] Completed job: {job}")
        job_queue.task_done()

workers = []
for i in range(3):
    t = threading.Thread(target=worker, args=(i,))
    t.start()
    workers.append(t)

submissions = [
    "Team A - PPT",
    "Team B - GitHub",
    "Team C - PPT",
    "Team D - GitHub",
    "Team E - PPT"
]

for submission in submissions:
    print(f"[API] Received submission: {submission}")
    job_queue.put(submission)

job_queue.join()

for _ in workers:
    job_queue.put(None)

for t in workers:
    t.join()

print("All submissions processed.")
