import asyncio
from runpod_flash import Endpoint
import os

os.environ["RUNPOD_API_KEY"]=""


async def main():

    ep = Endpoint(
        id="p4cz3lx70d1obs"
    )

    job = await ep.run({
        "action": "tts",
        "payload": {
            "text": "Xin chào, đây là bài test."
        }
    })

    print("JOB CREATED")
    print(job._data)

    await job.wait()

    print("JOB DONE")
    print(job._data)
    print(job.output)


asyncio.run(main())