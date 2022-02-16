from datetime import datetime


class Logger:
    def __init__(self, filename: str, mode: str = "a"):
        self.file = open(file=filename, mode=mode)

    def info(self, message: str) -> None:
        new_line = "INFO - {} - {}\n".format(datetime.now(), message)
        self._write(new_line)

    def _write(self, message: str) -> None:
        self.file.write(message)
        self.file.flush()
