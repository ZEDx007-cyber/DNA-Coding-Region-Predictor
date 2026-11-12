import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class WebAppTests(unittest.TestCase):
    def test_streamlit_app_renders_without_errors(self):
        app = AppTest.from_file(
            str(Path(__file__).parent / "web_app.py"), default_timeout=60
        ).run()

        self.assertEqual(
            [tab.label for tab in app.tabs],
            ["Predict", "Train", "Archive search", "About & data"],
        )
        self.assertEqual(len(app.exception), 0)


if __name__ == "__main__":
    unittest.main()
