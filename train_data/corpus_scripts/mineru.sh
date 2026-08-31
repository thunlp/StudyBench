source env_local.sh

# 把 raw_data/NBPhO 下所有 pdf 转为 markdown，结果写在每个 pdf 同名子目录中。
# 已存在 <pdf>/full.md 的会自动跳过；--force 可强制重跑。
python3 scripts/mineru_parse.py studybench_data/PhysicsBooks/quantum_physics \
    --model vlm --language en \
    --batch-size 20 --poll-interval 5 --is-ocr

# 调试用：先看会处理哪些文件
# python3 process/mineru_parse.py raw_data/NBPhO --dry-run

# 单文件试跑（限制 1 个）
# python3 process/mineru_parse.py raw_data/NBPhO --limit 1
