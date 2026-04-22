import json
import os
import re
import urllib.request

README_PATH = "README.md"

ACTIVITY_START = "<!--START_SECTION:activity-->"
ACTIVITY_END = "<!--END_SECTION:activity-->"

SUMMARY_START = "<!--START_SECTION:activity_summary-->"
SUMMARY_END = "<!--END_SECTION:activity_summary-->"

MODEL = "openai/gpt-4o-mini"
MODELS_URL = "https://models.github.ai/inference/chat/completions"


def extract_section(text: str, start: str, end: str) -> str:
    pattern = re.compile(rf"{re.escape(start)}(.*?){re.escape(end)}", re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise RuntimeError(f"Section markers not found: {start} ... {end}")
    return match.group(1).strip()


def replace_section(text: str, start: str, end: str, new_body: str) -> str:
    pattern = re.compile(rf"{re.escape(start)}.*?{re.escape(end)}", re.DOTALL)
    replacement = f"{start}\n{new_body}\n{end}"
    if not pattern.search(text):
        raise RuntimeError(f"Section markers not found: {start} ... {end}")
    return pattern.sub(replacement, text)


def normalize_activity_lines(raw: str):
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("<!--") and line.endswith("-->"):
            continue
        lines.append(line)
    return lines


def fallback_summary(lines):
    if not lines:
        return "- No recent public activity found."

    return "\n".join([
        "- Recent activity mainly centers on pull requests, issues, and discussion threads.",
        "- The latest items appear to focus on active open-source contribution rather than simple repository maintenance.",
        "- See the Recent Activity section below for the exact event list."
    ])


def call_github_models(activity_lines):
    prompt = "\n".join(activity_lines)

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are summarizing a developer's Recent Activity section for a GitHub profile README. "
                    "You will receive already-formatted recent activity lines. "
                    "Write 4 to 6 markdown bullet points. "
                    "Summarize the technical themes and concrete work. "
                    "For pull requests, explain what problem was addressed and how if the title makes that clear. "
                    "For issues, explain what problem was raised. "
                    "For comments or reviews, explain what topic was discussed. "
                    "Group related items when appropriate. "
                    "Do not invent details that are not reasonably supported by the activity lines. "
                    "Do not add a heading."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Summarize this Recent Activity section.\n\n"
                    "Requirements:\n"
                    "- 4 to 6 markdown bullets\n"
                    "- Focus on technical content, not event counts\n"
                    "- Merge related entries when helpful\n"
                    "- Be concise and concrete\n\n"
                    f"Recent Activity:\n{prompt}"
                ),
            },
        ],
        "temperature": 0.2,
        "max_tokens": 260,
    }

    req = urllib.request.Request(
        MODELS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}",
            "Content-Type": "application/json",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    return data["choices"][0]["message"]["content"].strip()


def main():
    with open(README_PATH, "r", encoding="utf-8") as f:
        readme = f.read()

    raw_activity = extract_section(readme, ACTIVITY_START, ACTIVITY_END)
    activity_lines = normalize_activity_lines(raw_activity)

    if not activity_lines:
        summary = "- No recent public activity found."
    else:
        try:
            summary = call_github_models(activity_lines)
            if not summary.strip():
                summary = fallback_summary(activity_lines)
        except Exception:
            summary = fallback_summary(activity_lines)

    if not summary.startswith("- "):
        summary = "- " + summary.lstrip("- ").strip()

    updated = replace_section(readme, SUMMARY_START, SUMMARY_END, summary)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated)


if __name__ == "__main__":
    main()
