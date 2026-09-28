# -*- coding: utf-8 -*-
"""
V6.4.1 scanner.py mojibake repair
目的：
1. 只修 scanner.py 中典型 UTF-8 -> Latin-1 mojibake。
2. 不修改選股公式、分數、gate、排序、Entry/Stop/RR 邏輯。
3. 修復後執行 AST syntax check。
4. 若仍偵測到典型亂碼，直接 FAIL，不允許當作完成版。
5. 自動保留 scanner.py.bak。
"""

from pathlib import Path
import ast
import re
import shutil
import sys

SRC = Path("scanner.py")
BAK = Path("scanner.py.bak")

# 典型 UTF-8 被誤當 Latin-1 後常見的起始字元。
MOJIBAKE_MARKERS = (
    "Ã", "Â", "â", "ð", "ï¼", "å", "æ", "ç", "é", "è", "ä",
)

# 已知 V6.4.1 正常繁中必須出現的代表性文字。
EXPECTED_TEXT = (
    "短期均線呈多頭排列",
    "中期趨勢維持向上",
    "成交量",
    "風險",
    "明日",
)

def looks_bad(line: str) -> bool:
    if any(m in line for m in MOJIBAKE_MARKERS):
        return True
    # C1 control chars are another strong mojibake signature.
    return any(0x80 <= ord(ch) <= 0x9F for ch in line)

def repair_once(s: str) -> str:
    """
    Classic mojibake reversal:
        UTF-8 bytes -> wrongly decoded as latin-1
    Reverse with:
        latin-1 encode -> UTF-8 decode
    """
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s

def repair_line(line: str) -> str:
    current = line
    # V6.4.1 observed corruption is normally one layer.
    # Allow up to 3 passes for nested corruption, but only keep a pass
    # when it reduces mojibake signatures.
    for _ in range(3):
        if not looks_bad(current):
            break
        candidate = repair_once(current)
        if candidate == current:
            break
        current = candidate
    return current

def bad_lines(text: str):
    out = []
    for no, line in enumerate(text.splitlines(), 1):
        if looks_bad(line):
            out.append((no, line))
    return out

def main():
    if not SRC.exists():
        raise SystemExit("ERROR: 找不到 scanner.py。請把本檔放在 scanner.py 同一層執行。")

    raw = SRC.read_text(encoding="utf-8-sig")

    before_bad = bad_lines(raw)
    print(f"[CHECK] 修復前疑似亂碼行數: {len(before_bad)}")

    if not BAK.exists():
        shutil.copy2(SRC, BAK)
        print(f"[BACKUP] {BAK}")

    fixed_lines = []
    changed = 0

    for line in raw.splitlines(keepends=True):
        new_line = repair_line(line)
        if new_line != line:
            changed += 1
        fixed_lines.append(new_line)

    fixed = "".join(fixed_lines)

    # Syntax gate: never write a syntactically broken scanner.py.
    try:
        ast.parse(fixed, filename="scanner.py")
    except SyntaxError as e:
        print(f"[FAIL] 修復後 Python syntax error: line {e.lineno}: {e.msg}")
        print("[SAFE] scanner.py 尚未覆蓋。")
        sys.exit(2)

    remain = bad_lines(fixed)
    if remain:
        print(f"[FAIL] 仍有 {len(remain)} 行疑似 mojibake，拒絕覆蓋 scanner.py。")
        for no, line in remain[:30]:
            print(f"  line {no}: {line[:180]!r}")
        print("[SAFE] scanner.py 尚未覆蓋。")
        sys.exit(3)

    # Ensure representative Chinese text survived / was recovered.
    found = [x for x in EXPECTED_TEXT if x in fixed]
    if len(found) < 3:
        print("[FAIL] 繁中內容驗證不足，拒絕覆蓋 scanner.py。")
        print("Found:", found)
        sys.exit(4)

    SRC.write_text(fixed, encoding="utf-8", newline="\n")

    # Read-back validation.
    verify = SRC.read_text(encoding="utf-8")
    ast.parse(verify, filename="scanner.py")

    remain2 = bad_lines(verify)
    if remain2:
        print("[FAIL] 寫回後仍偵測到亂碼。請立刻還原 scanner.py.bak")
        sys.exit(5)

    print(f"[OK] 修復行數: {changed}")
    print("[OK] Python syntax check passed")
    print("[OK] UTF-8 read-back passed")
    print("[OK] Mojibake scan passed")
    print("[OK] scanner.py 已以 UTF-8 寫回")
    print()
    print("下一步：python scanner.py")
    print("重新產生 docs/data.json 與 data/signals.csv。")

if __name__ == "__main__":
    main()
