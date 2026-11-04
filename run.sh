# 在终端执行 OLLAMA_CONTEXT_LENGTH=16384 ollama serve
# 另开一个终端，下载大模型并确认服务可用
# ollama pull qwen3:8b
curl http://localhost:11434/v1/models

# 运行评测
uv run python evals/run_agent_eval.py --out evals/reports/agent.json
