from spark.streaming_job import _wait_for_path
import os
import tempfile


class TestStreamingJob:
    def test_wait_for_path_returns_true_when_path_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            sub = os.path.join(tmp, "data")
            os.makedirs(sub)
            open(os.path.join(sub, "test.parquet"), "a").close()
            assert _wait_for_path(sub, max_wait=2)

    def test_wait_for_path_returns_false_when_path_missing(self):
        assert not _wait_for_path("/nonexistent/path", max_wait=2)
