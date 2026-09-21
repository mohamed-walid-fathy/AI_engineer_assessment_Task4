# Setup

Requirements: Python 3.10+.

```bash
pip install -r requirements.txt
```

## Windows (PowerShell)

```powershell
$env:GEMINI_API_KEY="your_key_here"   # or GOOGLE_API_KEY
python main.py --input dm_message_corpus_10k.json --output labels.xlsx

# No-cost / no-key tests
python main.py --input dm_message_corpus_10k.json --output dryrun.xlsx --dry-run --skip-sentiment
python main.py --input dm_message_corpus_10k.json --output trial.xlsx --limit 50 --skip-sentiment
```

## macOS / Linux (bash/zsh)

```bash
export GEMINI_API_KEY=your_key_here   # or GOOGLE_API_KEY
python main.py --input dm_message_corpus_10k.json --output labels.xlsx

# No-cost / no-key tests
python main.py --input dm_message_corpus_10k.json --output dryrun.xlsx --dry-run --skip-sentiment
python main.py --input dm_message_corpus_10k.json --output trial.xlsx --limit 50 --skip-sentiment
```

First run with sentiment enabled downloads ~1.1 GB (Hugging Face, one-time). Add `--skip-sentiment` to skip it.

See `README.md` for approach, output, and limitations. `labels.xlsx` is already included — no run required to evaluate.
