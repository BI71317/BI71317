import json
import os
import re
import urllib.request
from datetime import datetime, timedelta, timezone

OWNER = os.environ["GITHUB_REPOSITORY_OWNER"]
GH_TOKEN = os.environ["GITHUB_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

README_PATH = "README.md"
START = "<!--START_SECTION:monthly_summary-->"
END = "<!--END_SECTION:monthly_summary-->"

LOOKBACK_DAYS = 30


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


def openai_summary(prompt: str) -> str:
    body = {
        "model": "gpt-4.1-mini",
        "input": [
            {
                "role": "system",
                "content": (
                    "You summarize a developer's recent GitHub activity for a profile README. "
                    "Be concrete, concise, and professional. "
                    "Output markdown only. "
                    "Use 3-5 bullet points. "
                    "Focus on engineering themes, repositories, and kinds of contributions. "
                    "Do not invent work that is not supported by the activity data."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_output_tokens": 260,
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    parts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))

    return "\n".join(parts).strip()


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
            review = payload.get("review", {})
            pr = payload.get("pull_request", {})
            detail = f"{action} PR review on: {pr.get('title', '') or review.get('body', '')[:80]}"

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


def replace_section(readme: str, new_text: str) -> str:
    pattern = re.compile(
        rf"{re.escape(START)}.*?{re.escape(END)}",
        re.DOTALL,
    )
    replacement = f"{START}\n{new_text}\n{END}"

    if not pattern.search(readme):
        raise RuntimeError("monthly_summary markers not found in README.md")

    return pattern.sub(replacement, readme)


def build_fallback(events):
    if not events:
        return "- No significant public GitHub activity was detected in the last 30 days."

    repos = []
    event_types = {}

    for e in events:
        repo = e.get("repo", {}).get("name", "")
        et = e.get("type", "")
        if repo:
            repos.append(repo)
        event_types[et] = event_types.get(et, 0) + 1

    top_repos = []
    repo_count = {}
    for r in repos:
        repo_count[r] = repo_count.get(r, 0) + 1
    top_repos = sorted(repo_count.items(), key=lambda x: x[1], reverse=True)[:3]

    lines = ["- Public GitHub activity was recorded over the last 30 days."]
    if top_repos:
        repo_text = ", ".join([f"`{name}`" for name, _ in top_repos])
        lines.append(f"- Most visible activity appeared around {repo_text}.")
    if event_types:
        type_text = ", ".join([f"{k} ({v})" for k, v in sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:4]])
        lines.append(f"- Main event types included {type_text}.")
    return "\n".join(lines)


def main():
    events = gh_get(f"https://api.github.com/users/{OWNER}/events/public")
    recent_events = filter_recent_events(events, LOOKBACK_DAYS)

    if not recent_events:
        summary = "- No significant public GitHub activity was detected in the last 30 days."
    else:
        activity_text = format_events(recent_events)

        prompt = f"""
Summarize this developer's public GitHub activity from the last 30 days for a profile README.

Requirements:
- 3 to 5 bullets
- Focus on engineering themes, repositories, and types of contribution
- Prefer PRs, issues, fixes, reviews, debugging, implementation, and discussion over raw counts
- Mention repositories when possible
- Be concrete and conservative
- Avoid hype or generic praise
- Output markdown only

Activity:
{activity_text}
""".strip()

        try:
            summary = openai_summary(prompt)
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
