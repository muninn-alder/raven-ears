import os, unittest
from unittest.mock import patch

import lyrics_shared as LS


class _WithHotwords:
    def transcribe(self, audio, initial_prompt=None, hotwords=None): ...


class NamesDefaultTest(unittest.TestCase):
    def test_speech_default_is_prompt(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("RAVEN_EARS_NAMES_MODE", None)
            self.assertEqual(LS.names_mode(), "prompt")
            self.assertEqual(LS.whisper_name_kwargs(_WithHotwords(), ["Zibble"]), {"initial_prompt": "Zibble."})

    def test_hotwords_still_available_by_env(self):
        with patch.dict(os.environ, {"RAVEN_EARS_NAMES_MODE": "hotwords"}):
            self.assertEqual(LS.whisper_name_kwargs(_WithHotwords(), ["Zibble"]), {"hotwords": "Zibble"})


if __name__ == "__main__":
    unittest.main()
