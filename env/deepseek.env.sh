# DeepSeek settings for MechCAL development. No secrets here: the key comes from
# DEEPSEEK_API_KEY, which env/set_deepseek_key.sh stores in ~/.config/mechcal/secrets.env.
# Source this file (it is sourced automatically from ~/.bashrc after running the setup script).
# API reference: https://api-docs.deepseek.com/ (base URL without /v1; Responses API supported).

# Backbone used by agents and baselines (OpenAICompatibleSettings.from_env).
export MECHCAL_OPENAI_BASE_URL="https://api.deepseek.com"
export MECHCAL_OPENAI_MODEL="deepseek-flash"
export MECHCAL_OPENAI_API_KEY="${DEEPSEEK_API_KEY:-}"
export MECHCAL_OPENAI_API_PROTOCOL="responses"   # /responses supports json_schema outputs
export MECHCAL_OPENAI_REASONING_EFFORT="high"     # none | low | high | max
export MECHCAL_OPENAI_TIMEOUT="300"
export MECHCAL_OPENAI_MAX_RETRIES="2"

# Evidence judge. Development default: same model at maximum effort. To use a judge
# from another model family later, override MECHCAL_JUDGE_BASE_URL / _MODEL / _API_KEY.
export MECHCAL_JUDGE_BASE_URL="${MECHCAL_JUDGE_BASE_URL:-$MECHCAL_OPENAI_BASE_URL}"
export MECHCAL_JUDGE_MODEL="${MECHCAL_JUDGE_MODEL:-deepseek-flash}"
export MECHCAL_JUDGE_API_KEY="${MECHCAL_JUDGE_API_KEY:-${DEEPSEEK_API_KEY:-}}"
export MECHCAL_JUDGE_REASONING_EFFORT="${MECHCAL_JUDGE_REASONING_EFFORT:-max}"
export MECHCAL_JUDGE_TIMEOUT="${MECHCAL_JUDGE_TIMEOUT:-600}"
