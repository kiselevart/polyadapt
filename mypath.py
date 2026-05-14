import os

_KNOWN_DATASETS = {"ucf101", "hmdb51", "ucf10", "ucf11", "ssv2"}


class Path:
    @staticmethod
    def db_dir(database: str):
        db = (database or "").lower()
        if db not in _KNOWN_DATASETS:
            raise NotImplementedError(f"Dataset not available: {database}")
        data_root = os.getenv(
            "VNN_DATA_ROOT",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"),
        )
        root_dir   = os.path.join(data_root, db)
        output_dir = os.path.join(data_root, db + "_pre")
        return os.path.abspath(root_dir), os.path.abspath(output_dir)
