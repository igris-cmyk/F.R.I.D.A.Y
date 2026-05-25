import json
import unittest

from core.main import publish_failure_result


class FakeNATS:
    def __init__(self):
        self.published = []

    async def publish(self, subject, payload):
        self.published.append((subject, payload))


class TestRuntimeFailures(unittest.IsolatedAsyncioTestCase):
    async def test_publish_failure_result_emits_execution_result_event(self):
        nc = FakeNATS()

        await publish_failure_result(
            nc,
            trace_id="trace-123",
            error=RuntimeError("boom"),
            execution_time_ms=12,
        )

        self.assertEqual(len(nc.published), 1)
        subject, payload = nc.published[0]
        self.assertEqual(subject, "friday.stream.trace-123")

        event = json.loads(payload.decode())
        self.assertEqual(event["metadata"]["trace_id"], "trace-123")
        self.assertEqual(event["payload"]["status"], "failure")
        self.assertEqual(event["payload"]["error"], "boom")
        self.assertEqual(event["payload"]["execution_time_ms"], 12)


if __name__ == "__main__":
    unittest.main()
