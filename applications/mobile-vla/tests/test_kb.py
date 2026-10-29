import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from zhiman_kb import (  # noqa: E402
    add_entity,
    approve_workflow,
    connect,
    init_db,
    propose_workflow,
    redact_value,
    record_episode,
    rollback_workflow,
    search,
    summary,
)


class KnowledgeBaseTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "zhiman.db"
        self.conn = connect(self.db_path)
        init_db(self.conn)

    def tearDown(self):
        self.conn.close()
        self.temp_dir.cleanup()

    def test_review_gated_workflow_lifecycle(self):
        evidence = []
        for _ in range(3):
            evidence.append(
                record_episode(
                    self.conn,
                    task_key="send_arrival_message",
                    instruction="告诉女儿我到家了",
                    outcome="success",
                    risk_level="medium",
                    trace=[{"action": "open_app", "target": "测试通信应用"}],
                )
            )
        proposal = propose_workflow(
            self.conn,
            task_key="send_arrival_message",
            title="发送到家消息",
            body={"steps": ["打开应用", "选择联系人", "确认后发送"]},
            risk_level="medium",
            evidence_episode_ids=evidence,
        )
        self.assertEqual(search(self.conn, "到家")["workflows"], [])
        task_key, version = approve_workflow(self.conn, proposal, "人工复核通过")
        self.assertEqual(task_key, "send_arrival_message")
        self.assertEqual(version, 1)
        self.assertEqual(len(search(self.conn, "到家")["workflows"]), 1)

    def test_proposal_rejects_insufficient_evidence(self):
        episode = record_episode(
            self.conn,
            task_key="open_transit_code",
            instruction="打开乘车码",
            outcome="success",
            risk_level="low",
            trace=[],
        )
        with self.assertRaises(ValueError):
            propose_workflow(
                self.conn,
                task_key="open_transit_code",
                title="打开乘车码",
                body={"steps": []},
                risk_level="low",
                evidence_episode_ids=[episode],
            )

    def test_entities_are_retrievable(self):
        add_entity(self.conn, "女儿", "陈小雨", "测试通信应用", ["小雨"], None)
        result = search(self.conn, "女儿")
        self.assertEqual(result["entities"][0]["display_name"], "陈小雨")
        self.assertEqual(summary(self.conn)["entities"], 1)

    def test_sensitive_values_are_redacted_before_storage(self):
        episode_id = record_episode(
            self.conn,
            task_key="message",
            instruction="验证码 123456，手机号 13800138000",
            outcome="success",
            risk_level="medium",
            trace=[
                {
                    "verification_code": "654321",
                    "text": "身份证 320102199001011234",
                }
            ],
        )
        row = self.conn.execute(
            "SELECT instruction, trace_json FROM episodes WHERE id = ?", (episode_id,)
        ).fetchone()
        combined = row["instruction"] + row["trace_json"]
        self.assertNotIn("123456", combined)
        self.assertNotIn("13800138000", combined)
        self.assertNotIn("654321", combined)
        self.assertNotIn("320102199001011234", combined)
        self.assertIn("[REDACTED]", combined)

    def test_low_success_rate_blocks_proposal(self):
        evidence = []
        for _ in range(3):
            evidence.append(
                record_episode(
                    self.conn,
                    task_key="unstable_task",
                    instruction="不稳定任务",
                    outcome="success",
                    risk_level="low",
                    trace=[],
                )
            )
        for _ in range(2):
            record_episode(
                self.conn,
                task_key="unstable_task",
                instruction="不稳定任务",
                outcome="failure",
                risk_level="low",
                trace=[],
            )
        with self.assertRaises(ValueError):
            propose_workflow(
                self.conn,
                task_key="unstable_task",
                title="不稳定流程",
                body={"steps": []},
                risk_level="low",
                evidence_episode_ids=evidence,
            )

    def test_workflow_can_rollback_to_verified_version(self):
        def create_version(label):
            evidence = [
                record_episode(
                    self.conn,
                    task_key="font_size",
                    instruction="调整字体",
                    outcome="success",
                    risk_level="low",
                    trace=[{"label": label}],
                )
                for _ in range(3)
            ]
            proposal = propose_workflow(
                self.conn,
                task_key="font_size",
                title=label,
                body={"label": label},
                risk_level="low",
                evidence_episode_ids=evidence,
            )
            return approve_workflow(self.conn, proposal, f"批准{label}")[1]

        self.assertEqual(create_version("版本一"), 1)
        self.assertEqual(create_version("版本二"), 2)
        from_version, to_version = rollback_workflow(
            self.conn, "font_size", 1, "版本二出现回归"
        )
        self.assertEqual((from_version, to_version), (2, 1))
        row = self.conn.execute(
            "SELECT version FROM workflow_cards WHERE task_key='font_size' AND status='verified'"
        ).fetchone()
        self.assertEqual(row["version"], 1)
        audit = self.conn.execute(
            "SELECT event_type FROM workflow_audit ORDER BY id DESC LIMIT 1"
        ).fetchone()
        self.assertEqual(audit["event_type"], "rollback")

    def test_recursive_redaction(self):
        value = redact_value({"password": "secret", "nested": ["验证码 123456"]})
        self.assertEqual(value["password"], "[REDACTED]")
        self.assertNotIn("123456", value["nested"][0])


if __name__ == "__main__":
    unittest.main()
