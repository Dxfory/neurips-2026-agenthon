#!/usr/bin/env bash
# Local leak/privacy scan for the candidate tree. Prints paths/types only, never values.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
gitleaks dir --no-banner --redact --report-format json --report-path /tmp/agenthon_gl.json "$ROOT" >/dev/null 2>&1
python3 - <<'PY'
import json
d=json.load(open('/tmp/agenthon_gl.json'))
print(f"gitleaks findings: {len(d)}")
for f in d: print(f"  {f['File']}:{f['StartLine']} {f['RuleID']} (secret len {len(f['Secret'])})")
PY
echo "privacy patterns (file counts; scripts/ excluded):"
for p in 'university|大学|学院|学号|student[ _]id' '/Users/[A-Za-z]+|/home/[a-z]+' 'gmail\.com|outlook\.com|qq\.com|163\.com' '\+86|\+61' 'dealmesh|qbx' 'sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|AKIA[0-9A-Z]{16}'; do
  printf '  %-60s %s files\n' "$p" "$(rg -i -l -e "$p" "$ROOT" --glob '!scripts/*' --glob '!PLAN.md' 2>/dev/null | wc -l)"
done
echo "disallowed legal-name patterns (true positive if non-zero; scripts/ excluded):"
# Concatenated at runtime so this file does not store the spellings as literals.
legal_re="$(printf '%s%s|%s %s|%s %s|%s %s|%s%s' 'pei' 'dong' 'yiping' 'yin' 'xing' 'zheng' 'zheng' 'xing' 'zachar' 'iah')"
given_re="$(printf '%s%s%s' '\b' 'isa' 'ac\b')"
for p in "$legal_re" "$given_re" 'T1 Dev'; do
  printf '  %-60s %s files\n' "$p" "$(rg -i -l -e "$p" "$ROOT" --glob '!scripts/*' --glob '!PLAN.md' 2>/dev/null | wc -l)"
done
echo "team-handle / image-namespace patterns (expected; handles only):"
for p in 'yiping-yin|@Yiping-Yin' 'greedyshark' 'EscapedShark|@EscapedShark' '@Dxfory' 'dx/t4-'; do
  printf '  %-60s %s files\n' "$p" "$(rg -i -l -e "$p" "$ROOT" --glob '!scripts/*' --glob '!PLAN.md' 2>/dev/null | wc -l)"
done
