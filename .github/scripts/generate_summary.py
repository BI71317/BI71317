import os
import re
import json
import urllib.request
from datetime import datetime, timedelta, timezone

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage


OWNER = os.environ["GITHUB_REPOSITORY_OWNER"]
GH_TOKEN = os.environ["GITHUB_TOKEN"]  # for GitHub API
GITHUB_MODELS_TOKEN = os.environ["GITHUB_MODELS_TOKEN"]  # PAT for GitHub Models
README_PATH = "README.md"

START = "<!--START_SECTION:monthly_summary-->"
END = "<!--END_SECTION:monthly_summary-->"
LOOKBACK_DAYS = 30

MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o-mini"


def gh_get(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "monthly-summary-bot",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def parse_github_time(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def filter_recent_events(events, days: int):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return [e for e in events if parse_github_time(e["created_at"]) >= cutoff]


def format_events(events):
    lines = []

    for e in events[:50]:
        et = e.get("type", "")
        repo = e.get("repo", {}).get("name", "")
        created = e.get("created_at", "")
        payload = e.get("payload", {})

        detail = ""
        if et == "PushEvent":
            commits = payload.get("commits", [])
            msgs = [c.get("message", "").split("\n")[0] for c in commits[:3]]
            detail = f"{len(commits)} commit(s)"
            if msgs:
                detail += ": " + " | ".join(msgs)

        elif et == "PullRequestEvent":
            action = payload.get("action", "")
            pr = payload.get("pull_request", {})
            detail = f"{action} PR: {pr.get('title', '')}"

        elif et == "IssuesEvent":
            action = payload.get("action", "")
            issue = payload.get("issue", {})
            detail = f"{action} issue: {issue.get('title', '')}"

        elif et == "IssueCommentEvent":
            action = payload.get("action", "")
            issue = payload.get("issue", {})
            detail = f"{action} comment on: {issue.get('title', '')}"

        elif et == "PullRequestReviewEvent":
            action = payload.get("action", "")
            pr = payload.get("pull_request", {})
            detail = f"{action} PR review on: {pr.get('title', '')}"

        elif et == "CreateEvent":
            ref_type = payload.get("ref_type", "")
            ref = payload.get("ref", "")
            detail = f"created {ref_type}: {ref}"

        elif et == "ReleaseEvent":
            action = payload.get("action", "")
            rel = payload.get("release", {})
            detail = f"{action} release: {rel.get('name', '')}"

        else:
            detail = json.dumps(payload, ensure_ascii=False)[:160]

        lines.append(f"- {created} | {et} | {repo} | {detail}")

    return "\n".join(lines)


def build_fallback(events):
    if not events:
        return "- No significant public GitHub activity was detected in the last 30 days."

    repo_count = {}
    event_types = {}

    for e in events:
        repo = e.get("repo", {}).get("name", "")
        et = e.get("type", "")
        if repo:
            repo_count[repo] = repo_count.get(repo, 0) + 1
        event_types[et] = event_types.get(et, 0) + 1

    top_repos = sorted(repo_count.items(), key=lambda x: x[1], reverse=True)[:3]
    top_types = sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:4]

    lines = ["- Public GitHub activity was recorded over the last 30 days."]
    if top_repos:
        repo_text = ", ".join(f"`{name}`" for name, _ in top_repos)
        lines.append(f"- Most visible activity appeared around {repo_text}.")
    if top_types:
        type_text = ", ".join(f"{name} ({count})" for name, count in top_types)
        lines.append(f"- Main event types included {type_text}.")
    return "\n".join(lines)


def summarize_with_github_models(activity_text: str) -> str:
    client = ChatCompletionsClient(
        endpoint=MODELS_ENDPOINT,
        credential=AzureKeyCredential(GITHUB_MODELS_TOKEN),
        model=MODEL_NAME,
    )

    response = client.complete(
        messages=[
            SystemMessage(
                content=(
                    "You summarize a developer's recent GitHub activity for a profile README. "
                    "Be concrete, concise, and professional. "
                    "Output markdown only. "
                    "Use 3-5 bullet points. "
                    "Focus on engineering themes, repositories, and kinds of contributions. "
                    "Do not invent work that is not supported by the activity data."
                )
            ),
            UserMessage(
                content=(
                    "Summarize this developer's public GitHub activity from the last 30 days for a profile README.\n\n"
                    "Requirements:\n"
                    "- 3 to 5 bullets\n"
                    "- Focus on engineering themes, repositories, and types of contribution\n"
                    "- Prefer PRs, issues, fixes, reviews, debugging, implementation, and discussion over raw counts\n"
                    "- Mention repositories when possible\n"
                    "- Be concrete and conservative\n"
                    "- Avoid hype or generic praise\n"
                    "- Output markdown only\n\n"
                    f"Activity:\n{activity_text}"
                )
            ),
        ],
        temperature=0.2,
        max_tokens=260,
    )

    if not response.choices:
        return ""

    msg = response.choices[0].message
    content = getattr(msg, "content", None)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            text = getattr(item, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts).strip()

    return ""


def replace_section(readme: str, new_text: str) -> str:
    pattern = re.compile(rf"{re.escape(START)}.*?{re.escape(END)}", re.DOTALL)
    replacement = f"{START}\n{new_text}\n{END}"

    if not pattern.search(readme):
        raise RuntimeError("monthly_summary markers not found in README.md")

    return pattern.sub(replacement, readme)


def main():
    events = gh_get(f"https://api.github.com/users/{OWNER}/events/public")
    recent_events = filter_recent_events(events, LOOKBACK_DAYS)

    if not recent_events:
        summary = "- No significant public GitHub activity was detected in the last 30 days."
    else:
        activity_text = format_events(recent_events)
        try:
            summary = summarize_with_github_models(activity_text)
            if not summary:
                summary = build_fallback(recent_events)
        except Exception:
            summary = build_fallback(recent_events)

    with open(README_PATH, "r", encoding="utf-8") as f:
        readme = f.read()

    updated = replace_section(readme, summary)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated)


if __name__ == "__main__":
    main()
