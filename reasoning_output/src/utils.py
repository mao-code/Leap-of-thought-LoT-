import re

ANSWER_RE = re.compile(r"\*\*Answer:\*\*\s*(.+)", re.I)


def extract_answer(text: str) -> str:
    """Return the raw answer following the `**Answer:**` marker."""
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


def normalize_answer(ans: str) -> str:
    """Normalize numeric strings for robust comparison."""
    ans = ans.strip().strip(" .,!?")
    ans = ans.lstrip("$€£¥").replace(",", "")
    if re.fullmatch(r"-?\d+(?:\.\d+)?", ans):
        try:
            num = float(ans)
        except ValueError:
            return ans
        if num.is_integer():
            return str(int(num))
        ans = ("%f" % num).rstrip("0").rstrip(".")
    return ans
