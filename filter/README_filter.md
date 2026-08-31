# Guidance 构造流程

当前全量流程由 `filter/run_full_competition_pipeline.sh` 串联。它从竞赛题和教材出发，先抽取并规范化知识点，再检索、验证教材片段，最后由教师模型生成不泄露答案的解题指导。

运行入口：

```bash
bash filter/run_full_competition_pipeline.sh
```

需要先准备 `env_local.sh` 中的模型/API 配置。流程支持断点续跑：`START_AT=3` 从第 3 阶段开始，`DRY_RUN=1` 只打印命令。

## 前置：构建教材片段索引

| 脚本 | 输入 | 输出 |
| --- | --- | --- |
| `filter/build_textbook_fragments.py` | `train_data/PhysicsBooks/` 下的教材 Markdown | `studybench_data/textbook_fragments.jsonl` |

脚本按章节、段落和例题切分教材，保留 `fragment_id`、书名、章节、片段类型和原文。教材内容未变化时无需重复构建；总控脚本直接复用这个 JSONL 索引。

## 主流程

| 阶段 | 脚本 | 输入 | 输出 |
| --- | --- | --- | --- |
| 1. 抽取知识点 | `filter/extract_knowledge_points.py` | `studybench_data/competition_problems/competition_problems_full.json` | 在每个子题上写入 `knowledge_points` 字段，更新原文件 |
| 2. 规范化知识点 | `filter/canonicalize_knowledge_points.py` | 带 `knowledge_points` 的竞赛题库 | `studybench_data/knowledge_points.jsonl`、`knowledge_points.review.md` |
| 3. 检索候选片段 | `filter/retrieve_textbook_candidates.py` | `knowledge_points.jsonl`、`textbook_fragments.jsonl` | `studybench_data/kp_candidates.jsonl` |
| 4. 验证教材匹配 | `filter/verify_textbook_matches.py` | `kp_candidates.jsonl`、`textbook_fragments.jsonl` | `studybench_data/kp_matches.jsonl` |
| 5. 回填覆盖关系 | `filter/join_coverage.py` | 竞赛题库、`knowledge_points.jsonl`、`kp_matches.jsonl` | `studybench_data/competition_problems_full.with_coverage.json`，以及 coverage 报告/CSV |
| 6. 生成 guidance 并断点续跑 | `filter/build_kp_guidance.py` | 覆盖结果、`textbook_fragments.jsonl`、题库 skeleton、可选旧 guidance 文件 | `studybench_data/level3_guidance_full.json`、`studybench_data/level3_guidance_full.json.cache.jsonl` |

第 6 阶段对每个有教材覆盖的子题，把问题、参考解答、知识点和已验证教材片段交给教师模型。模型只输出解题方法和公式使用顺序，不应给出最终答案或关键数值计算。每次输出经过两道闸门：

1. 规则脱敏：`filter/build_targeted_guidance.py` 提供答案、结论和 `\\boxed{}` 等泄漏检测/清理函数。
2. LLM 检查：教师模型复核 guidance 是否泄露答案或代做关键计算；失败时自动重试，最终使用规则清理后的结果兜底。

`build_kp_guidance.py` 通过 `(source, year, source_problem_id, problem_id)` 缓存键断点续跑：先读取 JSONL cache，再从 `--resume-from` 指定的旧 guidance JSON 导入非空结果，最后只生成尚未命中的子题。默认旧 guidance 文件为 `eval/data/level3_guidance_full.json`，找不到时会跳过导入。
