"""Check that vector embedding does not bypass the limited content screen."""
import base64
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_repository import audit_embedded_media


class EmbeddedMediaTests(unittest.TestCase):
    def test_nested_vector_is_decoded(self):
        inner = '<svg xmlns="http://www.w3.org/2000/svg"><text>Relative concentration</text></svg>'
        encoded = base64.b64encode(inner.encode()).decode()
        outer = '<svg xmlns="http://www.w3.org/2000/svg"><image href="data:image/svg+xml,' + encoded + '"/></svg>'
        text = 'data:image/svg+xml,' + base64.b64encode(outer.encode()).decode()
        self.assertEqual(audit_embedded_media(text), 2)

    def test_embedded_machine_path_is_rejected(self):
        path = '/nfs' + '/scratch/private_example'
        vector = '<svg xmlns="http://www.w3.org/2000/svg"><text>' + path + '</text></svg>'
        text = 'data:image/svg+xml,' + base64.b64encode(vector.encode()).decode()
        with self.assertRaisesRegex(ValueError, 'machine path'):
            audit_embedded_media(text)


if __name__ == "__main__":
    unittest.main()
