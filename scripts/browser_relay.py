"""Tiny TCP relay: listen on a Windows loopback port, forward to a Docker-published port.

Why: with WSL `networkingMode=mirrored` (needed by AIML-04), Chrome/Edge time out on ports
published by Docker Desktop, while curl, PowerShell and Python reach them. The relay is an
ordinary Windows process, so the browser talks to it and it talks to Docker.

Usage (Windows): py -3.14 scripts/browser_relay.py 18501:8501 18000:8000
Then open http://127.0.0.1:18501 (dashboard) and http://127.0.0.1:18000/docs (scorer).
"""

from __future__ import annotations

import asyncio
import sys


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy bytes until EOF, then pass the half-close on (the other direction may still run)."""
    try:
        while data := await reader.read(65536):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except (ConnectionError, OSError):
        writer.close()


def handler(target_port: int, target_host: str = "127.0.0.1"):
    async def handle(c_reader: asyncio.StreamReader, c_writer: asyncio.StreamWriter) -> None:
        try:
            s_reader, s_writer = await asyncio.open_connection(target_host, target_port)
        except OSError:
            c_writer.close()
            return
        await asyncio.gather(pipe(c_reader, s_writer), pipe(s_reader, c_writer))
        s_writer.close()
        c_writer.close()

    return handle


async def main(pairs: list[str]) -> None:
    servers = []
    for pair in pairs:
        listen, target = (int(x) for x in pair.split(":"))
        servers.append(await asyncio.start_server(handler(target), "127.0.0.1", listen))
        print(f"relay http://127.0.0.1:{listen} -> 127.0.0.1:{target}", flush=True)
    await asyncio.gather(*(s.serve_forever() for s in servers))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:] or ["18501:8501", "18000:8000"]))
