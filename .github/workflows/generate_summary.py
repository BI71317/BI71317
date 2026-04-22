import json
import os
import re
import urllib.request

OWNER = os.environ["GITHUB_REPOSITORY_OWNER"]
REPO = os.environ["GITHUB_REPOSITORY"].split("/")[-1]
GH_TOKEN = os.environ["GITHUB_TOKEN"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

README_PATH = "README.md"
START = "<!--START_SECTION:weekly_summary-->"
END = "<!--END_SECTION:weekly_summary-->"

def gh_get(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "weekly-summary-bot",
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
                    "Do not invent work that is not supported by the activity data."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        "max_output_tokens": 220,
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

    # Responses API output text extraction
    parts = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                parts.append(content.get("text", ""))
    return "\n".join(parts).strip()

def format_events(events):
    lines = []
    for e in events[:20]:
        et = e.get("type", "")
        repo = e.get("repo", {}).get("name", "")
        created = e.get("created_at", "")
        payload = e.get("payload", {})

        detail = ""
        if et == "PushEvent":
            commits = payload.get("commits", [])
            msgs = [c.get("message", "").split("\n")[0] for c in commits[:3]]
            detail = f"{len(commits)} commit(s): " + " | ".join(msgs) if msgs else "push"
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
        elif et == "CreateEvent":
            ref_type = payload.get("ref_type", "")
            ref = payload.get("ref", "")
            detail = f"created {ref_type}: {ref}"
        elif et == "ReleaseEvent":
            action = payload.get("action", "")
            rel = payload.get("release", {})
            detail = f"{action} release: {rel.get('name', '')}"
        else:
            detail = json.dumps(payload)[:160]

        lines.append(f"- {created} | {et} | {repo} | {detail}")
    return "\n".join(lines)

def replace_section(readme: str, new_text: str) -> str:
    pattern = re.compile(
        rf"{re.escape(START)}.*?{re.escape(END)}",
        re.DOTALL
    )
    replacement = f"{START}\n{new_text}\n{END}"
    if not pattern.search(readme):
        raise RuntimeError("weekly_summary markers not found in README.md")
    return pattern.sub(replacement, readme)

def main():
    events = gh_get(f"https://api.github.com/users/{OWNER}/events/public")

    activity_text = format_events(events)
    prompt = f"""
Summarize this developer's recent public GitHub activity into a README section.

Requirements:
- 3 to 5 bullets
- Mention concrete repositories or themes when possible
- Prefer actual engineering themes over raw event counts
- Keep each bullet short
- Avoid fluff
- If activity is sparse, say so briefly and conservatively

Activity:
{activity_text}
""".strip()

    summary = openai_summary(prompt)

    with open(README_PATH, "r", encoding="utf-8") as f:
        readme = f.read()

    updated = replace_section(readme, summary)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated)

if __name__ == "__main__":
    main()
