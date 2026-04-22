import os
import re
import json
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Tuple

from azure.ai.inference import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential
from azure.ai.inference.models import SystemMessage, UserMessage

OWNER = os.environ["GITHUB_REPOSITORY_OWNER"]
GH_TOKEN = os.environ["GITHUB_TOKEN"]                  # GitHub API
GITHUB_MODELS_TOKEN = os.environ["GITHUB_MODELS_TOKEN"]  # GitHub Models PAT

README_PATH = "README.md"
START = "<!--START_SECTION:activity_details-->"
END = "<!--END_SECTION:activity_details-->"

MODELS_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL_NAME = "gpt-4o-mini"

MAX_EVENTS = 10


def gh_get(url: str):
    req = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {GH_TOKEN}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "activity-details-bot",
        },
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))


def gh_get_safe(url: str):
    try:
        return gh_get(url)
    except (urllib.error.HTTPError, urllib.error.URLError):
        return None


def truncate(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def replace_section(readme: str, new_text: str) -> str:
    pattern = re.compile(rf"{re.escape(START)}.*?{re.escape(END)}", re.DOTALL)
    replacement = f"{START}\n{new_text}\n{END}"
    if not pattern.search(readme):
        raise RuntimeError("activity_details markers not found in README.md")
    return pattern.sub(replacement, readme)


def extract_issue_or_pr_url(event: Dict) -> Optional[str]:
    payload = event.get("payload", {})

    if event.get("type") in {"PullRequestEvent", "PullRequestReviewEvent"}:
        pr = payload.get("pull_request", {})
        return pr.get("url")

    if event.get("type") in {"IssuesEvent", "IssueCommentEvent"}:
        issue = payload.get("issue", {})
        return issue.get("url")

    return None


def fetch_issue_detail(issue_api_url: str) -> Optional[Dict]:
    issue = gh_get_safe(issue_api_url)
    if not issue:
        return None

    comments_preview = []
    comments_url = issue.get("comments_url")
    if comments_url and issue.get("comments", 0) > 0:
        comments = gh_get_safe(comments_url) or []
        if isinstance(comments, list):
            for c in comments[:2]:
                comments_preview.append({
                    "user": c.get("user", {}).get("login", ""),
                    "body": truncate(c.get("body", ""), 220),
                })

    return {
        "kind": "issue",
        "repo": issue.get("repository_url", "").replace("https://api.github.com/repos/", ""),
        "number": issue.get("number"),
        "title": issue.get("title", ""),
        "state": issue.get("state", ""),
        "labels": [x.get("name", "") for x in issue.get("labels", [])[:5]],
        "body": truncate(issue.get("body", ""), 500),
        "comments_preview": comments_preview,
        "html_url": issue.get("html_url", ""),
    }


def fetch_pr_detail(pr_api_url: str) -> Optional[Dict]:
    pr = gh_get_safe(pr_api_url)
    if not pr:
        return None

    commits_preview = []
    commits_url = pr.get("commits_url")
    commits = gh_get_safe(commits_url) or []
    if isinstance(commits, list):
        for c in commits[:3]:
            commits_preview.append(truncate(c.get("commit", {}).get("message", ""), 160))

    review_comments_preview = []
    review_comments_url = pr.get("review_comments_url", "").replace("{/number}", "")
    if review_comments_url:
        review_comments = gh_get_safe(review_comments_url) or []
        if isinstance(review_comments, list):
            for rc in review_comments[:2]:
                review_comments_preview.append({
                    "user": rc.get("user", {}).get("login", ""),
                    "path": rc.get("path", ""),
                    "body": truncate(rc.get("body", ""), 220),
                })

    return {
        "kind": "pull_request",
        "repo": pr.get("base", {}).get("repo", {}).get("full_name", ""),
        "number": pr.get("number"),
        "title": pr.get("title", ""),
        "state": pr.get("state", ""),
        "merged": bool(pr.get("merged")),
        "body": truncate(pr.get("body", ""), 550),
        "changed_files": pr.get("changed_files", 0),
        "additions": pr.get("additions", 0),
        "deletions": pr.get("deletions", 0),
        "commits_preview": commits_preview,
        "review_comments_preview": review_comments_preview,
        "html_url": pr.get("html_url", ""),
    }


def fetch_comment_detail(comment_url: str) -> Optional[Dict]:
    comment = gh_get_safe(comment_url)
    if not comment:
        return None
    return {
        "user": comment.get("user", {}).get("login", ""),
        "body": truncate(comment.get("body", ""), 320),
        "html_url": comment.get("html_url", ""),
    }


def describe_event_data(event: Dict) -> Dict:
    et = event.get("type", "")
    created_at = event.get("created_at", "")
    repo = event.get("repo", {}).get("name", "")
    payload = event.get("payload", {})

    result = {
        "event_type": et,
        "created_at": created_at,
        "repo": repo,
        "action": payload.get("action", ""),
    }

    if et == "PushEvent":
        commits = payload.get("commits", [])
        result["commits"] = [
            truncate(c.get("message", "").split("\n")[0], 140)
            for c in commits[:4]
        ]
        result["ref"] = payload.get("ref", "")
        return result

    if et == "CreateEvent":
        result["ref_type"] = payload.get("ref_type", "")
        result["ref"] = payload.get("ref", "")
        return result

    if et == "DeleteEvent":
        result["ref_type"] = payload.get("ref_type", "")
        result["ref"] = payload.get("ref", "")
        return result

    if et == "PullRequestEvent":
        pr_url = payload.get("pull_request", {}).get("url")
        result["pr"] = fetch_pr_detail(pr_url) if pr_url else None
        return result

    if et == "PullRequestReviewEvent":
        pr_url = payload.get("pull_request", {}).get("url")
        review = payload.get("review", {})
        review_url = review.get("url")
        result["pr"] = fetch_pr_detail(pr_url) if pr_url else None
        result["review"] = fetch_comment_detail(review_url) if review_url else {
            "user": review.get("user", {}).get("login", ""),
            "body": truncate(review.get("body", ""), 320),
        }
        return result

    if et == "IssuesEvent":
        issue_url = payload.get("issue", {}).get("url")
        result["issue"] = fetch_issue_detail(issue_url) if issue_url else None
        return result

    if et == "IssueCommentEvent":
        issue_url = payload.get("issue", {}).get("url")
        comment_url = payload.get("comment", {}).get("url")
        result["issue"] = fetch_issue_detail(issue_url) if issue_url else None
        result["comment"] = fetch_comment_detail(comment_url) if comment_url else {
            "user": payload.get("comment", {}).get("user", {}).get("login", ""),
            "body": truncate(payload.get("comment", {}).get("body", ""), 320),
        }
        return result

    return result


def fallback_summary(data: Dict) -> str:
    et = data.get("event_type", "")
    repo = data.get("repo", "")

    if et == "PushEvent":
        commits = data.get("commits", [])
        if commits:
            return f"- Pushed updates to `{repo}`, including: {', '.join(commits[:2])}."
        return f"- Pushed updates to `{repo}`."

    if et == "CreateEvent":
        return f"- Created a {data.get('ref_type', 'ref')} in `{repo}`: `{data.get('ref', '')}`."

    if et == "DeleteEvent":
        return f"- Deleted a {data.get('ref_type', 'ref')} in `{repo}`: `{data.get('ref', '')}`."

    if et == "PullRequestEvent":
        pr = data.get("pr")
        if pr:
            if pr.get("merged"):
                return f"- Merged PR #{pr['number']} in `{pr['repo']}`: {pr['title']}."
            return f"- Updated PR #{pr['number']} in `{pr['repo']}`: {pr['title']}."
        return f"- Worked on a pull request in `{repo}`."

    if et == "PullRequestReviewEvent":
        pr = data.get("pr")
        if pr:
            return f"- Reviewed PR #{pr['number']} in `{pr['repo']}`: {pr['title']}."
        return f"- Reviewed a pull request in `{repo}`."

    if et == "IssuesEvent":
        issue = data.get("issue")
        if issue:
            return f"- Worked on issue #{issue['number']} in `{issue['repo']}`: {issue['title']}."
        return f"- Worked on an issue in `{repo}`."

    if et == "IssueCommentEvent":
        issue = data.get("issue")
        if issue:
            return f"- Commented on issue #{issue['number']} in `{issue['repo']}`: {issue['title']}."
        return f"- Commented on an issue in `{repo}`."

    return f"- Recorded `{et}` activity in `{repo}`."


def summarize_one_event(client: ChatCompletionsClient, data: Dict) -> str:
    prompt = json.dumps(data, ensure_ascii=False, indent=2)

    response = client.complete(
        messages=[
            SystemMessage(
                content=(
                    "You are writing a GitHub profile README section. "
                    "Summarize exactly one recent GitHub activity item in one concise markdown bullet. "
                    "Be concrete. Mention the repository. "
                    "For a PR, explain what problem it addressed and how, using the title/body/commits conservatively. "
                    "For a merged PR, explicitly say it was merged. "
                    "For an issue, explain what problem was raised. "
                    "For a discussion/comment/review, explain what topic was discussed. "
                    "For a push, summarize the change based on commit messages. "
                    "Do not invent details not supported by the input."
                )
            ),
            UserMessage(
                content=(
                    "Write one markdown bullet for this single GitHub activity item.\n\n"
                    f"{prompt}"
                )
            ),
        ],
        temperature=0.2,
        max_tokens=140,
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


def main():
    events = gh_get(f"https://api.github.com/users/{OWNER}/events/public?per_page={MAX_EVENTS}")

    client = ChatCompletionsClient(
        endpoint=MODELS_ENDPOINT,
        credential=AzureKeyCredential(GITHUB_MODELS_TOKEN),
        model=MODEL_NAME,
    )

    bullets = []
    for event in events[:MAX_EVENTS]:
        data = describe_event_data(event)
        try:
            line = summarize_one_event(client, data)
            if not line:
                line = fallback_summary(data)
        except Exception:
            line = fallback_summary(data)

        if not line.startswith("- "):
            line = "- " + line.lstrip("- ").strip()

        bullets.append(line)

    new_text = "\n".join(bullets) if bullets else "- No recent public activity found."

    with open(README_PATH, "r", encoding="utf-8") as f:
        readme = f.read()

    updated = replace_section(readme, new_text)

    with open(README_PATH, "w", encoding="utf-8") as f:
        f.write(updated)


if __name__ == "__main__":
    main()
