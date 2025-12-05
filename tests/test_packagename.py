import shutil
from pathlib import Path

import pytest


class TestPipeline:
    @pytest.fixture(autouse=True)
    def copy_config_folder(self):
        config_folder = Path("packagename_configs")
        if not config_folder.exists():
            shutil.copytree(
                "../packagename_configs",
                config_folder,
            )

    @pytest.fixture(autouse=True)
    def change_test_dir(self, request, monkeypatch):
        test_dir = request.fspath.dirname
        monkeypatch.chdir(request.fspath.dirname)
        return test_dir
    
    def test_clean_up(self):
        self.clean_folder()

    def clean_folder(self):
        paths = [
            "experiments/",
        ]
        import os

        for f in paths:
            if "/" not in f and os.path.exists(f):
                os.remove(f)
            else:
                shutil.rmtree(f, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def cleanup(request):
    """Cleanup a testing directory once we are finished."""

    def remove_test_dir():
        paths = [
            "experiments/",
            "packagename_configs/",
        ]
        import os

        for f in paths:
            if "/" not in f and os.path.exists(f):
                os.remove(f)
            else:
                shutil.rmtree(
                    f"{os.path.dirname(os.path.realpath(__file__))}/{f}", ignore_errors=True
                )

    request.addfinalizer(remove_test_dir)
