import json
import random
from pathlib import Path
from string import Template

TEMPLATE_PATH = Path("input_prompt/prompts/leap_templates.json")

def load_templates():
    return json.loads(TEMPLATE_PATH.read_text())

TEMPLATES = load_templates()

def build_prompt(question: str, fewshot: bool = False, leap: bool = True):
    # tpl = random.choice(TEMPLATES)
    tpl = TEMPLATES[0]

    tpl_msg = Template(tpl["leap_template"]) if leap else Template(tpl["non_leap_template"])
    user_msg = tpl_msg.substitute(question=question)

    msgs = []
    if fewshot and tpl.get("fewshot"):
        msgs.extend(tpl["fewshot"])  # list of dicts with role/content
    msgs.append({"role": "user", "content": user_msg})
    return msgs

def build_plain_prompt(question: str, fewshot: bool = False, leap: bool = True) -> str:
    # pick the first template (you could random.choice if you want)
    tpl = TEMPLATES[0]

    # choose between leap- and non-leap template
    tpl_msg = Template(tpl["leap_template"]) if leap else Template(tpl["non_leap_template"])
    user_msg = tpl_msg.substitute(question=question)

    # collect all text‐chunks (fewshot examples + the final user prompt)
    parts = []
    if fewshot and "fewshot" in tpl:
        for msg in tpl["fewshot"]:
            # each msg is like {"role": "...", "content": "..."}
            parts.append(msg["content"].strip())

    # finally append the "user" content
    parts.append(user_msg.strip())

    # join everything with two newlines (or whatever separator you prefer)
    return "\n\n".join(parts)