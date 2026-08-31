source env_local.sh

# Textbook corpus: PDF -> Markdown -> light OCR cleanup -> eval redaction.
# Each PDF gets a same-named subdirectory with MinerU's full.md, cleaned
# ocrfix.md, and ocrfix.redacted.md. Existing outputs are skipped; pass
# --force on a step to rerun.

# 1) MinerU parse (VLM + OCR)
python3 corpus_scripts/mineru_parse.py PhysicsBooks \
    --model vlm --language en \
    --batch-size 20 --poll-interval 5 --is-ocr

# 2) Clean full.md: drop leaked VLM prompts and rejoin split digits in math.
python3 corpus_scripts/ocr_fix.py PhysicsBooks

# 3) Redact ocrfix.md spans that also appear in the eval JSON for that book.
#    Last folder name under PhysicsBooks must match the JSON "source" field.
#    Writes ocrfix.redacted.md next to each ocrfix.md; does not change the JSON.
python3 corpus_scripts/redact_eval.py eval/data/qwen3_8b_textbook_problem.json \
    --books-dir PhysicsBooks
