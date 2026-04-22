import os
import re
import json
import urllib.request
import urllib.error
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage


OWNER = os.environ["GITHUB_REPOSITORY_OWNER"]
GH_TOKEN = os.environ["GITHUB_TOKEN"]  # GitHub API
GITHUB_MODELS_TOKEN = os.environ["GITHUB_MODELS_TOKEN"]  # GitHub Models PAT

README_PATH = "README.md"
START = "<!--START_SECTION:monthly_summary-->"
END = "<!--END_SECTION:monthly_summary-->"

LOOKBACK_DAYS = 30
MAX_EVENTS = 100
MAX_ENRICHED_ITEMS = 16

MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o-mini"


def gh_get(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "monthly-summary-bot",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def gh_get_safe(url: str):
    try:
        return gh_get(url)
    except urllib.error.HTTPError:
        return None
    except urllib.error.URLError:
        return None


def parse_github_time(ts: str) -> datetime:
    return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)


def filter_recent_events(events, days: int):
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    return [e for e in events if parse_github_time(e["created_at"]) >= cutoff]


def truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def extract_repo_and_number_from_api_url(url: str) -> Optional[Tuple[str, int, str]]:
    """
    Examples:
    https://api.github.com/repos/OWNER/REPO/issues/123
    https://api.github.com/repos/OWNER/REPO/pulls/456
    Returns: ("OWNER/REPO", 123, "issues"|"pulls")
    """
    m = re.search(r"/repos/([^/]+/[^/]+)/(issues|pulls)/(\d+)$", url)
    if not m:
        return None
    repo, kind, number = m.group(1), m.group(2), int(m.group(3))
    return repo, number, kind


def fetch_issue_detail(issue_api_url: str) -> Optional[Dict]:
    issue = gh_get_safe(issue_api_url)
    if not issue:
        return None

    comments_data = []
    comments_url = issue.get("comments_url")
    if comments_url and issue.get("comments", 0) > 0:
        comments_data = gh_get_safe(comments_url) or []
        if isinstance(comments_data, list):
            comments_data = comments_data[:2]
        else:
            comments_data = []

    return {
        "kind": "issue",
        "repo": issue.get("repository_url", "").replace("https://api.github.com/repos/", ""),
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        "state": issue.get("state", ""),
        "body": truncate(issue.get("body", ""), 400),
        "labels": [lbl.get("name", "") for lbl in issue.get("labels", [])[:5]],
        "comments": [
            {
                "user": c.get("user", {}).get("login", ""),
                "body": truncate(c.get("body", ""), 220),
            }
            for c in comments_data
            if isinstance(c, dict)
        ],
        "html_url": issue.get("html_url", ""),
    }


def fetch_pull_detail(pull_api_url: str) -> Optional[Dict]:
    pr = gh_get_safe(pull_api_url)
    if not pr:
        return None

    review_summary = []
    reviews_url = pr.get("review_comments_url", "").replace("{/number}", "")
    if reviews_url:
        review_comments = gh_get_safe(reviews_url) or []
        if isinstance(review_comments, list):
            for rc in review_comments[:2]:
                if isinstance(rc, dict):
                    review_summary.append(
                        {
                            "user": rc.get("user", {}).get("login", ""),
                            "body": truncate(rc.get("body", ""), 220),
                            "path": rc.get("path", ""),
                        }
                    )

    commits_summary = []
    commits_url = pr.get("commits_url")
    if commits_url:
        commits = gh_get_safe(commits_url) or []
        if isinstance(commits, list):
            for c in commits[:3]:
                if isinstance(c, dict):
                    commits_summary.append(
                        truncate(c.get("commit", {}).get("message", ""), 140)
                    )

    return {
        "kind": "pull_request",
        "repo": pr.get("base", {}).get("repo", {}).get("full_name", ""),
        "number": pr.get("number"),
        "title": pr.get("title", ""),
        "state": pr.get("state", ""),
        "merged": pr.get("merged", False),
        "body": truncate(pr.get("body", ""), 450),
        "additions": pr.get("additions", 0),
        "deletions": pr.get("deletions", 0),
        "changed_files": pr.get("changed_files", 0),
        "commits": commits_summary,
        "review_comments": review_summary,
        "html_url": pr.get("html_url", ""),
    }


def format_basic_event_line(e: Dict) -> str:
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
        detail = f"created {payload.get('ref_type', '')}: {payload.get('ref', '')}"
    elif et == "DeleteEvent":
        detail = f"deleted {payload.get('ref_type', '')}: {payload.get('ref', '')}"
    else:
        detail = truncate(json.dumps(payload, ensure_ascii=False), 180)

    return f"- {created} | {et} | {repo} | {detail}"


def collect_enriched_items(events: List[Dict]) -> List[Dict]:
    items: List[Dict] = []
    seen_keys = set()

    for e in events:
        et = e.get("type", "")
        payload = e.get("payload", {})

        # PullRequestEvent / PullRequestReviewEvent
        if et in {"PullRequestEvent", "PullRequestReviewEvent"}:
            pr = payload.get("pull_request", {})
            pr_url = pr.get("url")
            if pr_url:
                parsed = extract_repo_and_number_from_api_url(pr_url)
                if parsed:
                    repo, number, _ = parsed
                    key = ("pr", repo, number)
                    if key not in seen_keys:
                        detail = fetch_pull_detail(pr_url)
                        if detail:
                            items.append(detail)
                            seen_keys.add(key)

        # IssuesEvent / IssueCommentEvent
        if et in {"IssuesEvent", "IssueCommentEvent"}:
            issue = payload.get("issue", {})
            issue_url = issue.get("url")
            if issue_url:
                parsed = extract_repo_and_number_from_api_url(issue_url)
                if parsed:
                    repo, number, _ = parsed
                    key = ("issue", repo, number)
                    if key not in seen_keys:
                        detail = fetch_issue_detail(issue_url)
                        if detail:
                            items.append(detail)
                            seen_keys.add(key)

        if len(items) >= MAX_ENRICHED_ITEMS:
            break

    return items[:MAX_ENRICHED_ITEMS]


def format_enriched_items(items: List[Dict]) -> str:
    lines = []

    for item in items:
        if item["kind"] == "pull_request":
            lines.append(
                f"- PR | {item['repo']}#{item['number']} | "
                f"title={item['title']} | state={item['state']} | merged={item['merged']} | "
                f"files={item['changed_files']} | +{item['additions']}/-{item['deletions']}"
            )
            if item.get("body"):
                lines.append(f"  body: {item['body']}")
            if item.get("commits"):
                lines.append(f"  commits: {' || '.join(item['commits'])}")
            if item.get("review_comments"):
                for rc in item["review_comments"][:2]:
                    lines.append(
                        f"  review_comment by {rc['user']} on {rc['path']}: {rc['body']}"
                    )

        elif item["kind"] == "issue":
            labels = ", ".join(item.get("labels", []))
            lines.append(
                f"- ISSUE | {item['repo']}#{item['number']} | "
                f"title={item['title']} | state={item['state']} | labels={labels}"
            )
            if item.get("body"):
                lines.append(f"  body: {item['body']}")
            if item.get("comments"):
                for c in item["comments"][:2]:
                    lines.append(f"  comment by {c['user']}: {c['body']}")

    return "\n".join(lines)


def build_fallback(events: List[Dict], enriched_items: List[Dict]) -> str:
    if enriched_items:
        bullets = []
        pr_items = [x for x in enriched_items if x["kind"] == "pull_request"][:2]
        issue_items = [x for x in enriched_items if x["kind"] == "issue"][:2]

        for pr in pr_items:
            bullets.append(
                f"- Worked on `{pr['repo']}` through PR #{pr['number']} ({pr['title']})."
            )
        for issue in issue_items:
            bullets.append(
                f"- Discussed or tracked issues in `{issue['repo']}` such as #{issue['number']} ({issue['title']})."
            )

        if bullets:
            return "\n".join(bullets[:4])

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


def summarize_with_github_models(basic_events_text: str, enriched_text: str) -> str:
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
                    "Prefer actual technical topics over raw counts. "
                    "Do not invent work that is not supported by the activity data."
                )
            ),
            UserMessage(
                content=(
                    "Summarize this developer's public GitHub activity from the last 30 days for a profile README.\n\n"
                    "Requirements:\n"
                    "- 3 to 5 bullets\n"
                    "- Mention specific repositories when possible\n"
                    "- Prefer actual technical themes, bug areas, implementation topics, PRs, issues, reviews, and debugging work\n"
                    "- Avoid generic praise, avoid event-count summaries, avoid saying only 'public activity was recorded'\n"
                    "- If the data suggests compiler, GPU, codegen, correctness, CI, build, or repo-maintenance themes, mention them conservatively\n"
                    "- If evidence is weak, stay conservative\n"
                    "- Output markdown only\n\n"
                    "Basic event stream:\n"
                    f"{basic_events_text}\n\n"
                    "Enriched issue / PR details:\n"
                    f"{enriched_text}"
                )
            ),
        ],
        temperature=0.2,
        max_tokens=320,
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
    events = gh_get(
        f"https://api.github.com/users/{OWNER}/events/public?per_page={MAX_EVENTS}"
    )
    recent_events = filter_recent_events(events, LOOKBACK_DAYS)

    if not recent_events:
        summary = "- No significant public GitHub activity was detected in the last 30 days."
    else:
        basic_events_text = "\n".join(format_basic_event_line(e) for e in recent_events[:40])
        enriched_items = collect_enriched_items(recent_events)
        enriched_text = format_enriched_items(enriched_items)

        try:
            summary = summarize_with_github_models(basic_events_text, enriched_text)
            if not summary or "Public GitHub activity was recorded" in summary:
                summary = build_fallback(recent_events, enriched_items)
        except Exception:
            summary = build_fallback(recent_events, enriched_items)

    with open(README_PATH, "r", encoding="utf-8") as f:
        readme = f.read()

    updated = replace_section(readme, summary)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated)


if __name__ == "__main__":
    main()
