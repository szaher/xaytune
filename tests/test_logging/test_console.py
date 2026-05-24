from xaytune.logging.console import ConsoleBackend


class TestConsoleBackend:
    def test_log_scalar(self, capsys):
        backend = ConsoleBackend()
        backend.log_scalar("loss", 0.5432, 10)
        captured = capsys.readouterr()
        assert "loss" in captured.out
        assert "0.5432" in captured.out
        assert "10" in captured.out

    def test_log_config(self, capsys):
        backend = ConsoleBackend()
        backend.log_config({"lr": 0.001, "epochs": 3})
        captured = capsys.readouterr()
        assert "lr" in captured.out
        assert "0.001" in captured.out

    def test_close_is_noop(self):
        backend = ConsoleBackend()
        backend.close()  # should not raise

    def test_multiple_scalars_formatted(self, capsys):
        backend = ConsoleBackend()
        backend.log_scalar("loss", 0.5, 1)
        backend.log_scalar("lr", 0.001, 1)
        captured = capsys.readouterr()
        assert "loss" in captured.out
        assert "lr" in captured.out

    def test_is_logging_backend(self):
        from xaytune.logging.base import LoggingBackend

        backend = ConsoleBackend()
        assert isinstance(backend, LoggingBackend)
