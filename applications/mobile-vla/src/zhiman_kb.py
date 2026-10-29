"""Local, review-gated knowledge base for the Zhiman mobile assistant."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


SCHEMA_VERSION = 2
RISK_LEVELS = {"low", "medium", "high"}
OUTCOMES = {"success", "failure", "aborted"}
FORBIDDEN_KEYS = {
    "password",
    "payment_pin",
    "verification_code",
    "id_number",
    "bank_card_number",
}
SENSITIVE_PATTERNS = (
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), "[PHONE_REDACTED]"),
    (re.compile(r"(?<!\d)\d{17}[0-9Xx](?!\d)"), "[ID_REDACTED]"),
    (re.compile(r"(?<!\d)(?:\d[ -]?){16,19}(?!\d)"), "[CARD_REDACTED]"),
    (re.compile(r"(?<!\d)\d{6}(?!\d)"), "[CODE_REDACTED]"),
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            relation TEXT NOT NULL,
            display_name TEXT NOT NULL,
            preferred_channel TEXT,
            aliases_json TEXT NOT NULL DEFAULT '[]',
            note TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS episodes (
            id TEXT PRIMARY KEY,
            task_key TEXT NOT NULL,
            instruction TEXT NOT NULL,
            app_package TEXT,
            started_at TEXT NOT NULL,
            ended_at TEXT NOT NULL,
            outcome TEXT NOT NULL CHECK(outcome IN ('success', 'failure', 'aborted')),
            risk_level TEXT NOT NULL CHECK(risk_level IN ('low', 'medium', 'high')),
            trace_json TEXT NOT NULL,
            failure_reason TEXT,
            user_feedback TEXT,
            source_device TEXT,
            redacted INTEGER NOT NULL CHECK(redacted IN (0, 1))
        );

        CREATE TABLE IF NOT EXISTS knowledge_proposals (
            id TEXT PRIMARY KEY,
            task_key TEXT NOT NULL,
            title TEXT NOT NULL,
            body_json TEXT NOT NULL,
            risk_level TEXT NOT NULL CHECK(risk_level IN ('low', 'medium', 'high')),
            evidence_episode_ids_json TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('proposed', 'approved', 'rejected')),
            created_at TEXT NOT NULL,
            reviewed_at TEXT,
            reviewer_note TEXT
        );

        CREATE TABLE IF NOT EXISTS workflow_cards (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_key TEXT NOT NULL,
            version INTEGER NOT NULL,
            title TEXT NOT NULL,
            body_json TEXT NOT NULL,
            risk_level TEXT NOT NULL CHECK(risk_level IN ('low', 'medium', 'high')),
            status TEXT NOT NULL CHECK(status IN ('verified', 'deprecated')),
            source_proposal_id TEXT NOT NULL REFERENCES knowledge_proposals(id),
            created_at TEXT NOT NULL,
            UNIQUE(task_key, version)
        );

        CREATE TABLE IF NOT EXISTS workflow_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_key TEXT NOT NULL,
            event_type TEXT NOT NULL CHECK(event_type IN ('approve', 'rollback')),
            from_version INTEGER,
            to_version INTEGER NOT NULL,
            reviewer_note TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_episodes_task_outcome
            ON episodes(task_key, outcome);
        CREATE INDEX IF NOT EXISTS idx_workflows_task_status
            ON workflow_cards(task_key, status);
        """
    )
    conn.execute(
        "INSERT OR REPLACE INTO metadata(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()


def parse_json_text(value: str) -> Any:
    candidate = Path(value)
    if candidate.is_file():
        return json.loads(candidate.read_text(encoding="utf-8"))
    return json.loads(value)


def redact_text(value: str) -> str:
    result = value
    for pattern, replacement in SENSITIVE_PATTERNS:
        result = pattern.sub(replacement, result)
    return result


def redact_value(value: Any, key: str | None = None) -> Any:
    if key and key.lower() in FORBIDDEN_KEYS:
        return "[REDACTED]"
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, list):
        return [redact_value(item) for item in value]
    if isinstance(value, dict):
        return {item_key: redact_value(item, item_key) for item_key, item in value.items()}
    return value


def add_entity(
    conn: sqlite3.Connection,
    relation: str,
    display_name: str,
    preferred_channel: str | None,
    aliases: list[str],
    note: str | None,
) -> int:
    cursor = conn.execute(
        """
        INSERT INTO entities(
            relation, display_name, preferred_channel, aliases_json, note, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            relation,
            display_name,
            preferred_channel,
            json.dumps(aliases, ensure_ascii=False),
            note,
            utc_now(),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def record_episode(
    conn: sqlite3.Connection,
    *,
    task_key: str,
    instruction: str,
    outcome: str,
    risk_level: str,
    trace: Any,
    app_package: str | None = None,
    failure_reason: str | None = None,
    user_feedback: str | None = None,
    source_device: str | None = None,
    episode_id: str | None = None,
) -> str:
    if outcome not in OUTCOMES:
        raise ValueError(f"unsupported outcome: {outcome}")
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"unsupported risk level: {risk_level}")
    episode_id = episode_id or str(uuid.uuid4())
    now = utc_now()
    safe_instruction = redact_text(instruction)
    safe_trace = redact_value(trace)
    safe_failure_reason = redact_text(failure_reason) if failure_reason else None
    safe_feedback = redact_text(user_feedback) if user_feedback else None
    conn.execute(
        """
        INSERT INTO episodes(
            id, task_key, instruction, app_package, started_at, ended_at,
            outcome, risk_level, trace_json, failure_reason, user_feedback,
            source_device, redacted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
        """,
        (
            episode_id,
            task_key,
            safe_instruction,
            app_package,
            now,
            now,
            outcome,
            risk_level,
            json.dumps(safe_trace, ensure_ascii=False),
            safe_failure_reason,
            safe_feedback,
            source_device,
        ),
    )
    conn.commit()
    return episode_id


def propose_workflow(
    conn: sqlite3.Connection,
    *,
    task_key: str,
    title: str,
    body: Any,
    risk_level: str,
    evidence_episode_ids: Iterable[str],
    minimum_successes: int = 3,
    minimum_success_rate: float = 0.8,
) -> str:
    if risk_level not in RISK_LEVELS:
        raise ValueError(f"unsupported risk level: {risk_level}")
    evidence = list(dict.fromkeys(evidence_episode_ids))
    if len(evidence) < minimum_successes:
        raise ValueError(f"at least {minimum_successes} successful episodes are required")
    placeholders = ",".join("?" for _ in evidence)
    rows = conn.execute(
        f"SELECT id, task_key, outcome, redacted FROM episodes WHERE id IN ({placeholders})",
        evidence,
    ).fetchall()
    if len(rows) != len(evidence):
        raise ValueError("one or more evidence episodes do not exist")
    if any(
        row["task_key"] != task_key
        or row["outcome"] != "success"
        or row["redacted"] != 1
        for row in rows
    ):
        raise ValueError("all evidence must be redacted successful episodes for the same task")
    task_stats = conn.execute(
        """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN outcome = 'success' THEN 1 ELSE 0 END) AS successes
        FROM episodes WHERE task_key = ?
        """,
        (task_key,),
    ).fetchone()
    total = int(task_stats["total"])
    successes = int(task_stats["successes"] or 0)
    success_rate = successes / total if total else 0.0
    if success_rate < minimum_success_rate:
        raise ValueError(
            f"task success rate {success_rate:.3f} is below {minimum_success_rate:.3f}"
        )
    proposal_id = str(uuid.uuid4())
    conn.execute(
        """
        INSERT INTO knowledge_proposals(
            id, task_key, title, body_json, risk_level,
            evidence_episode_ids_json, status, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, 'proposed', ?)
        """,
        (
            proposal_id,
            task_key,
            title,
            json.dumps(body, ensure_ascii=False),
            risk_level,
            json.dumps(evidence, ensure_ascii=False),
            utc_now(),
        ),
    )
    conn.commit()
    return proposal_id


def approve_workflow(
    conn: sqlite3.Connection, proposal_id: str, reviewer_note: str
) -> tuple[str, int]:
    proposal = conn.execute(
        "SELECT * FROM knowledge_proposals WHERE id = ?", (proposal_id,)
    ).fetchone()
    if proposal is None:
        raise ValueError("proposal does not exist")
    if proposal["status"] != "proposed":
        raise ValueError("proposal has already been reviewed")
    row = conn.execute(
        "SELECT COALESCE(MAX(version), 0) AS max_version FROM workflow_cards WHERE task_key = ?",
        (proposal["task_key"],),
    ).fetchone()
    version = int(row["max_version"]) + 1
    now = utc_now()
    previous = conn.execute(
        "SELECT version FROM workflow_cards WHERE task_key = ? AND status = 'verified'",
        (proposal["task_key"],),
    ).fetchone()
    conn.execute(
        "UPDATE workflow_cards SET status = 'deprecated' WHERE task_key = ? AND status = 'verified'",
        (proposal["task_key"],),
    )
    conn.execute(
        """
        INSERT INTO workflow_cards(
            task_key, version, title, body_json, risk_level,
            status, source_proposal_id, created_at
        ) VALUES (?, ?, ?, ?, ?, 'verified', ?, ?)
        """,
        (
            proposal["task_key"],
            version,
            proposal["title"],
            proposal["body_json"],
            proposal["risk_level"],
            proposal_id,
            now,
        ),
    )
    conn.execute(
        """
        UPDATE knowledge_proposals
        SET status = 'approved', reviewed_at = ?, reviewer_note = ?
        WHERE id = ?
        """,
        (now, reviewer_note, proposal_id),
    )
    conn.execute(
        """
        INSERT INTO workflow_audit(
            task_key, event_type, from_version, to_version, reviewer_note, created_at
        ) VALUES (?, 'approve', ?, ?, ?, ?)
        """,
        (
            proposal["task_key"],
            int(previous["version"]) if previous else None,
            version,
            reviewer_note,
            now,
        ),
    )
    conn.commit()
    return proposal["task_key"], version


def rollback_workflow(
    conn: sqlite3.Connection, task_key: str, target_version: int, reviewer_note: str
) -> tuple[int | None, int]:
    target = conn.execute(
        "SELECT version FROM workflow_cards WHERE task_key = ? AND version = ?",
        (task_key, target_version),
    ).fetchone()
    if target is None:
        raise ValueError("target workflow version does not exist")
    current = conn.execute(
        "SELECT version FROM workflow_cards WHERE task_key = ? AND status = 'verified'",
        (task_key,),
    ).fetchone()
    from_version = int(current["version"]) if current else None
    if from_version == target_version:
        raise ValueError("target workflow version is already active")
    now = utc_now()
    conn.execute(
        "UPDATE workflow_cards SET status = 'deprecated' WHERE task_key = ?",
        (task_key,),
    )
    conn.execute(
        "UPDATE workflow_cards SET status = 'verified' WHERE task_key = ? AND version = ?",
        (task_key, target_version),
    )
    conn.execute(
        """
        INSERT INTO workflow_audit(
            task_key, event_type, from_version, to_version, reviewer_note, created_at
        ) VALUES (?, 'rollback', ?, ?, ?, ?)
        """,
        (task_key, from_version, target_version, reviewer_note, now),
    )
    conn.commit()
    return from_version, target_version


def search(conn: sqlite3.Connection, query: str) -> dict[str, list[dict[str, Any]]]:
    like = f"%{query}%"
    entities = conn.execute(
        """
        SELECT id, relation, display_name, preferred_channel, aliases_json, note
        FROM entities
        WHERE relation LIKE ? OR display_name LIKE ? OR aliases_json LIKE ? OR note LIKE ?
        ORDER BY updated_at DESC
        LIMIT 20
        """,
        (like, like, like, like),
    ).fetchall()
    workflows = conn.execute(
        """
        SELECT task_key, version, title, body_json, risk_level
        FROM workflow_cards
        WHERE status = 'verified'
          AND (task_key LIKE ? OR title LIKE ? OR body_json LIKE ?)
        ORDER BY task_key, version DESC
        LIMIT 20
        """,
        (like, like, like),
    ).fetchall()
    return {
        "entities": [dict(row) for row in entities],
        "workflows": [dict(row) for row in workflows],
    }


def summary(conn: sqlite3.Connection) -> dict[str, int]:
    result: dict[str, int] = {}
    for table in (
        "entities",
        "episodes",
        "knowledge_proposals",
        "workflow_cards",
        "workflow_audit",
    ):
        result[table] = int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    result["verified_workflows"] = int(
        conn.execute(
            "SELECT COUNT(*) FROM workflow_cards WHERE status = 'verified'"
        ).fetchone()[0]
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, required=True)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("init")
    sub.add_parser("summary")

    entity = sub.add_parser("add-entity")
    entity.add_argument("--relation", required=True)
    entity.add_argument("--name", required=True)
    entity.add_argument("--channel")
    entity.add_argument("--alias", action="append", default=[])
    entity.add_argument("--note")

    episode = sub.add_parser("record-episode")
    episode.add_argument("--task-key", required=True)
    episode.add_argument("--instruction", required=True)
    episode.add_argument("--outcome", choices=sorted(OUTCOMES), required=True)
    episode.add_argument("--risk", choices=sorted(RISK_LEVELS), required=True)
    episode.add_argument("--trace", required=True, help="JSON text or JSON file")
    episode.add_argument("--app-package")
    episode.add_argument("--failure-reason")
    episode.add_argument("--user-feedback")
    episode.add_argument("--source-device")

    proposal = sub.add_parser("propose-workflow")
    proposal.add_argument("--task-key", required=True)
    proposal.add_argument("--title", required=True)
    proposal.add_argument("--body", required=True, help="JSON text or JSON file")
    proposal.add_argument("--risk", choices=sorted(RISK_LEVELS), required=True)
    proposal.add_argument("--evidence-id", action="append", required=True)

    approval = sub.add_parser("approve-workflow")
    approval.add_argument("--proposal-id", required=True)
    approval.add_argument("--reviewer-note", required=True)

    rollback = sub.add_parser("rollback-workflow")
    rollback.add_argument("--task-key", required=True)
    rollback.add_argument("--target-version", type=int, required=True)
    rollback.add_argument("--reviewer-note", required=True)

    search_parser = sub.add_parser("search")
    search_parser.add_argument("--query", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    with connect(args.db) as conn:
        init_db(conn)
        if args.command == "init":
            payload: Any = {"schema_version": SCHEMA_VERSION, "db": str(args.db)}
        elif args.command == "summary":
            payload = summary(conn)
        elif args.command == "add-entity":
            payload = {
                "entity_id": add_entity(
                    conn,
                    args.relation,
                    args.name,
                    args.channel,
                    args.alias,
                    args.note,
                )
            }
        elif args.command == "record-episode":
            payload = {
                "episode_id": record_episode(
                    conn,
                    task_key=args.task_key,
                    instruction=args.instruction,
                    outcome=args.outcome,
                    risk_level=args.risk,
                    trace=parse_json_text(args.trace),
                    app_package=args.app_package,
                    failure_reason=args.failure_reason,
                    user_feedback=args.user_feedback,
                    source_device=args.source_device,
                )
            }
        elif args.command == "propose-workflow":
            payload = {
                "proposal_id": propose_workflow(
                    conn,
                    task_key=args.task_key,
                    title=args.title,
                    body=parse_json_text(args.body),
                    risk_level=args.risk,
                    evidence_episode_ids=args.evidence_id,
                )
            }
        elif args.command == "approve-workflow":
            task_key, version = approve_workflow(
                conn, args.proposal_id, args.reviewer_note
            )
            payload = {"task_key": task_key, "version": version}
        elif args.command == "rollback-workflow":
            from_version, to_version = rollback_workflow(
                conn, args.task_key, args.target_version, args.reviewer_note
            )
            payload = {"from_version": from_version, "to_version": to_version}
        elif args.command == "search":
            payload = search(conn, args.query)
        else:
            raise AssertionError(args.command)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
