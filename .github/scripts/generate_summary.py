import os
import re

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

README_PATH = "README.md"

ACTIVITY_START = "<!--START_SECTION:activity-->"
ACTIVITY_END = "<!--END_SECTION:activity-->"

SUMMARY_START = "<!--START_SECTION:activity_summary-->"
SUMMARY_END = "<!--END_SECTION:activity_summary-->"

MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o-mini"
GITHUB_MODELS_TOKEN = os.environ["GITHUB_MODELS_TOKEN"]


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
        return "- No recent activity found."

    bullets = []
    for line in lines[:5]:
        clean = re.sub(r"\s+", " ", line).strip()
        bullets.append(f"- {clean}")
    return "\n".join(bullets)


def summarize_activity(lines):
    if not lines:
        return "- No recent activity found."

    activity_text = "\n".join(lines)

    client = ChatCompletionsClient(
        endpoint=MODELS_ENDPOINT,
        credential=AzureKeyCredential(GITHUB_MODELS_TOKEN),
        model=MODEL_NAME,
    )

    response = client.complete(
        messages=[
            SystemMessage(
                content=(
                    "You are summarizing a developer's recent GitHub activity for a profile README. "
                    "You will be given the already-generated 'Recent Activity' lines. "
                    "Write 4 to 6 concise markdown bullet points. "
                    "Group related items when helpful. "
                    "Infer the topic from the activity text conservatively. "
                    "For pull requests, explain what problem was addressed and how if the title makes that clear. "
                    "For issues, explain what problem was raised. "
                    "For discussions/comments/reviews, explain what topic was discussed. "
                    "Do not invent details beyond what is reasonably supported by the activity lines."
                )
            ),
            UserMessage(
                content=(
                    "Summarize the following Recent Activity section into a short README summary.\n\n"
                    "Requirements:\n"
                    "- 4 to 6 markdown bullets\n"
                    "- Prefer technical themes over raw repetition\n"
                    "- Mention repositories when useful\n"
                    "- Merge related PR/issue/comment activity into coherent bullets when appropriate\n"
                    "- Avoid repeating the same repo name in every line unless necessary\n"
                    "- Do not output a heading\n\n"
                    f"Recent Activity:\n{activity_text}"
                )
            ),
        ],
        temperature=0.2,
        max_tokens=260,
    )

    if not response.choices:
        return fallback_summary(lines)

    msg = response.choices[0].message
    content = getattr(msg, "content", None)

    if isinstance(content, str):
        text = content.strip()
        return text if text else fallback_summary(lines)

    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        joined = "\n".join(parts).strip()
        return joined if joined else fallback_summary(lines)

    return fallback_summary(lines)


def main():
    with open(README_PATH, "r", encoding="utf-8") as f:
        readme = f.read()

    raw_activity = extract_section(readme, ACTIVITY_START, ACTIVITY_END)
    activity_lines = normalize_activity_lines(raw_activity)

    summary = summarize_activity(activity_lines)

    if not summary.startswith("- "):
        summary = "- " + summary.lstrip("- ").strip()

    updated = replace_section(readme, SUMMARY_START, SUMMARY_END, summary)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated)


if __name__ == "__main__":
    main()
