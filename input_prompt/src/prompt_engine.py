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
