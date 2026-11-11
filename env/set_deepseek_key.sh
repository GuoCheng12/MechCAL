#!/usr/bin/env bash
# One-time setup: store the DeepSeek API key outside the repository and load it in new shells.
# Usage:  bash env/set_deepseek_key.sh      (the key is read without echo; never pass it as an argument)
set -euo pipefail

repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dir="$HOME/.config/mechcal"
secrets="$dir/secrets.env"

read -r -s -p "DeepSeek API key: " key
echo
if [[ -z "$key" || "$key" =~ [[:space:]] ]]; then
  echo "No key stored: the key is empty or contains whitespace." >&2
  exit 1
fi

mkdir -p "$dir"
chmod 700 "$dir"
umask 077
printf "export DEEPSEEK_API_KEY=%q\n" "$key" > "$secrets"
chmod 600 "$secrets"
unset key

marker="# >>> mechcal deepseek >>>"
if ! grep -qF "$marker" "$HOME/.bashrc" 2>/dev/null; then
  cat >> "$HOME/.bashrc" <<EOF
$marker
[ -f "$secrets" ] && . "$secrets"
[ -f "$repo/env/deepseek.env.sh" ] && . "$repo/env/deepseek.env.sh"
# <<< mechcal deepseek <<<
EOF
fi

echo "Stored in $secrets (mode 600). New shells load it automatically;"
echo "for this shell run:  source $secrets && source $repo/env/deepseek.env.sh"
