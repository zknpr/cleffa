"""Stock baseline decodes wire envelopes without patching model execution."""
import base64
import copy
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'bench'))
import omni_pytorch_latency as benchmark


class StockRequest(unittest.TestCase):
    def test_encoded_video_keeps_exact_bytes_and_input_ownership(self):
        raw=b'\0\0\0\x08ftypfixture'
        request={'videos':[{'content_type':'video/quicktime','base64':base64.b64encode(raw).decode()}]}
        original=copy.deepcopy(request)
        with benchmark.stock_request(request,'clef-flash') as decoded:
            path=Path(decoded['videos'][0]);self.assertEqual(path.read_bytes(),raw)
        self.assertFalse(path.exists())
        with benchmark.stock_request(request,'clef-omni') as decoded:
            self.assertEqual(decoded['videos'],[raw])
        self.assertEqual(request,original)


if __name__=='__main__':unittest.main()
