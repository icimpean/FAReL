import csv


class LogEntry(object):
    """A log entry containing data for experiments.

    Attributes:
        entry_fields: The names of the (column) fields used for each entry.
    """
    def __init__(self):
        self.entry_fields = []
        self.path = None

    def create_entry(self, *args):
        """Method to create an entry for the log."""
        raise NotImplementedError

    def create_file(self, path, from_checkpoint=False):
        """Create the CSV file"""
        if not from_checkpoint:
            with open(path, "w", newline="") as file_writer:
                writer = csv.DictWriter(file_writer, fieldnames=self.entry_fields)
                writer.writeheader()
        self.path = path

    def write_data(self, data, path=None):
        """Write entries to a given file."""
        if path is None:
            path = self.path
        if not isinstance(data, list):
            data = [data]
        with open(path, "a", newline="") as file_writer:
            writer = csv.DictWriter(file_writer, fieldnames=self.entry_fields)
            writer.writerows(data)
