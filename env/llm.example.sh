# Set credentials in your shell or in an ignored env/llm.local.sh file.
export MECHCAL_OPENAI_BASE_URL="https://your-provider.example/v1"
export MECHCAL_OPENAI_MODEL="your-model"
export MECHCAL_OPENAI_API_KEY="your-api-key"
export MECHCAL_OPENAI_TEMPERATURE="0"
export MECHCAL_OPENAI_TOP_P="1"
export MECHCAL_OPENAI_REASONING_EFFORT="omit"
export MECHCAL_OPENAI_MAX_TOKENS="4800"
export MECHCAL_OPENAI_TIMEOUT="120"
export MECHCAL_OPENAI_MAX_RETRIES="2"
export MECHCAL_OPENAI_TRUST_ENV="false"

export MECHCAL_SUPPORT_JUDGE_BASE_URL="https://your-judge-provider.example/v1"
export MECHCAL_SUPPORT_JUDGE_MODEL="your-judge-model"
export MECHCAL_SUPPORT_JUDGE_API_KEY="your-judge-api-key"
export MECHCAL_SUPPORT_JUDGE_TIMEOUT="120"

# Optional, when Amesp is installed and authorized locally.
# export MECHCAL_AMESP_BIN="/path/to/amesp"
# export MECHCAL_AMESP_TIMEOUT="300"
# export MECHCAL_AMESP_NPARA="1"
# export MECHCAL_AMESP_MAXCORE_MB="1000"
