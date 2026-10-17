"""Tests for the ppt-to-md / xls-to-md / epub-to-md / audio-transcribe actions
and the --out-dir option of utils_run.py (stdlib unittest).

    .venv/bin/python -m unittest discover -s test -p 'test_convert_actions.py' -v

Fixtures: test/fixtures/convert/ (regenerate with make_fixtures.py). The .ppt
and .xls variants are produced from them by LibreOffice inside each test. The
audio test is slow (Whisper + pyannote) → only runs with UTILS_TEST_AUDIO=1.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUN = os.path.join(ROOT, 'utils_run.py')
FIX = os.path.join(ROOT, 'test', 'fixtures', 'convert')
sys.path.insert(0, os.path.join(ROOT, 'actions', 'audio_utils'))


def run(*args) -> tuple:
    p = subprocess.run([sys.executable, RUN, *args], capture_output=True, text=True, timeout=900)
    return p.returncode, json.loads(p.stdout)


def lo_convert(src, fmt, outdir):
    subprocess.run(['libreoffice', '--headless', '--convert-to', fmt, '--outdir', outdir, src],
                   check=True, capture_output=True, timeout=300)
    return os.path.join(outdir, os.path.splitext(os.path.basename(src))[0] + '.' + fmt)


class ConvertActions(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix='utils_convert_test_')
        self.src = os.path.join(self.tmp, 'src')
        self.out = os.path.join(self.tmp, 'out')
        os.mkdir(self.src)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _copy(self, name):
        dst = os.path.join(self.src, name)
        shutil.copy(os.path.join(FIX, name), dst)
        return dst

    def _convert(self, path, action):
        before = sorted(os.listdir(self.src))
        code, res = run(path, action, '--out-dir', self.out)
        self.assertEqual(code, 0, res)
        self.assertEqual(res['status'], 'ok')
        self.assertEqual(os.path.dirname(res['output_file']), self.out)
        self.assertEqual(sorted(os.listdir(self.src)), before, 'input folder must stay untouched')
        self.assertEqual([f for f in os.listdir(self.out) if not f.endswith('.md')], [],
                         'staged input must be removed from --out-dir')
        with open(res['output_file'], encoding='utf-8') as f:
            return f.read()

    # --- registry / contract -------------------------------------------------
    def test_list_all_has_new_actions(self):
        code, actions = run('--list-all')
        self.assertEqual(code, 0)
        names = {a['action'] for a in actions}
        self.assertTrue({'ppt-to-md', 'xls-to-md', 'epub-to-md', 'audio-transcribe'} <= names)

    def test_default_output_beside_input(self):
        path = self._copy('sample.epub')
        code, res = run(path, 'epub-to-md')
        self.assertEqual(res, {'status': 'ok', 'output_file': os.path.join(self.src, 'sample.md')})

    def test_out_dir_refuses_different_file_with_same_name(self):
        path = self._copy('sample.epub')
        os.makedirs(self.out)
        open(os.path.join(self.out, 'sample.epub'), 'w').close()
        code, res = run(path, 'epub-to-md', '--out-dir', self.out)
        self.assertEqual(code, 1)
        self.assertIn('already contains', res['message'])

    # --- ppt-to-md -----------------------------------------------------------
    def _check_ppt(self, md):
        for n in range(1, 7):
            self.assertIn(f'<!-- slide {n} -->', md)
        self.assertLess(md.index('<!-- slide 2 -->'), md.index('<!-- slide 3 -->'))
        self.assertIn('## Slide 1 — Corporate Strategy', md)
        self.assertIn('  - Economies of scale', md)
        self.assertIn('<!-- notes slide 1 -->', md)
        self.assertIn('Start with the Porter quote.', md)
        self.assertIn('| Beta \\| Gamma | 25% |', md)
        self.assertIn('| 2024 | 12.5 |', md)
        self.assertIn('[image-only slide]', md)
        self.assertIn('Backup slide (hidden)', md)

    def test_pptx(self):
        self._check_ppt(self._convert(self._copy('sample.pptx'), 'ppt-to-md'))

    def test_ppt_via_libreoffice(self):
        ppt = lo_convert(os.path.join(FIX, 'sample.pptx'), 'ppt', self.src)
        self._check_ppt(self._convert(ppt, 'ppt-to-md'))

    # --- xls-to-md -----------------------------------------------------------
    def _check_xls(self, md):
        self.assertIn('<!-- sheet 1: P&L -->', md)
        self.assertIn('| | A | B | C |', md)
        self.assertIn('| 4 | Margin | 40 | 50.5 |', md)
        self.assertIn('- `B4`: `=B2-B3` → 40', md)
        self.assertIn('- `C6`: `=C2/B2-1` → 0.205', md)
        self.assertIn('Growth \\| YoY', md)
        self.assertIn('## Sheet 3 — Scratch (hidden)', md)
        self.assertIn('[empty sheet]', md)

    def test_xlsx(self):
        self._check_xls(self._convert(self._copy('sample.xlsx'), 'xls-to-md'))

    def test_xlsm(self):
        self._check_xls(self._convert(self._copy('sample.xlsm'), 'xls-to-md'))

    def test_xls_via_libreoffice(self):
        xls = lo_convert(os.path.join(FIX, 'sample.xlsx'), 'xls', self.src)
        self._check_xls(self._convert(xls, 'xls-to-md'))

    # --- epub-to-md ----------------------------------------------------------
    def test_epub(self):
        md = self._convert(self._copy('sample.epub'), 'epub-to-md')
        self.assertIn('# Sample Strategy Book', md)
        self.assertIn('<!-- chapter 2 -->\n## Competitive Advantage', md)
        self.assertIn('### Cost leadership', md)
        self.assertNotIn('<!-- chapter 4 -->', md)

    # --- audio-transcribe ----------------------------------------------------
    def test_srt_to_markdown(self):
        import transcribe_to_md as t
        srt = ('1\n00:00:01,000 --> 00:00:02,000\nSpeaker 1: Hello.\n\n'
               '2\n00:00:02,500 --> 00:00:03,000\nSpeaker 1: Welcome.\n\n'
               '3\n01:02:03,400 --> 01:02:05,000\nSpeaker 2: Note: a question.\n')
        md = t.to_markdown(t.parse_srt(srt), 'x')
        self.assertIn('[00:00:01] **Speaker 1**: Hello. Welcome.', md)
        self.assertIn('[01:02:03] **Speaker 2**: Note: a question.', md)

    @unittest.skipUnless(os.environ.get('UTILS_TEST_AUDIO'), 'slow: set UTILS_TEST_AUDIO=1')
    def test_audio_transcribe(self):
        path = self._copy('sample_dialogue.mp3')
        code, res = run(path, 'audio-transcribe', '--out-dir', self.out,
                        '--model', 'turbo', '--max-speakers', '2')
        self.assertEqual(code, 0, res)
        self.assertEqual(os.path.basename(res['output_file']), 'sample_dialogue_transcription.md')
        self.assertEqual(sorted(os.listdir(self.out)), ['sample_dialogue_transcription.md'])
        self.assertEqual(os.listdir(self.src), ['sample_dialogue.mp3'])
        with open(res['output_file'], encoding='utf-8') as f:
            md = f.read()
        self.assertRegex(md, r'\[00:00:\d\d\] \*\*Speaker 1\*\*: ')
        self.assertIn('competitive advantage', md.lower())


if __name__ == '__main__':
    unittest.main()
