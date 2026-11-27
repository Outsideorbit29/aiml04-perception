"""scripts/browser_relay.py forwards a TCP connection end to end."""

import asyncio

from scripts.browser_relay import handler


def test_relay_forwards_bytes():
    async def run():
        async def echo(reader, writer):
            writer.write(b"pong:" + await reader.read(100))
            await writer.drain()
            writer.close()

        backend = await asyncio.start_server(echo, "127.0.0.1", 0)
        port = backend.sockets[0].getsockname()[1]
        relay = await asyncio.start_server(handler(port), "127.0.0.1", 0)
        rport = relay.sockets[0].getsockname()[1]
        reader, writer = await asyncio.open_connection("127.0.0.1", rport)
        writer.write(b"ping")
        await writer.drain()
        writer.write_eof()
        data = await asyncio.wait_for(reader.read(100), 5)
        writer.close()
        relay.close()
        backend.close()
        return data

    assert asyncio.run(run()) == b"pong:ping"
