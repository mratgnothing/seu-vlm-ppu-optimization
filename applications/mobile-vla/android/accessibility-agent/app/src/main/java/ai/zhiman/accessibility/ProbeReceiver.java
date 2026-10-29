package ai.zhiman.accessibility;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

import org.json.JSONObject;

import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

public final class ProbeReceiver extends BroadcastReceiver {
    public static final String ACTION_PROBE = "ai.zhiman.accessibility.SEMANTIC_PROBE";
    public static final String ACTION_ARM_OFFLINE_PROOF =
            "ai.zhiman.accessibility.ARM_OFFLINE_PROOF";
    public static final String ACTION_ARM_OFFLINE_IME_PROOF =
            "ai.zhiman.accessibility.ARM_OFFLINE_IME_PROOF";
    public static final String ACTION_ARM_COMMUNICATION_PLANNER_PROOF =
            "ai.zhiman.accessibility.ARM_COMMUNICATION_PLANNER_PROOF";

    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent == null) {
            return;
        }
        if (ACTION_ARM_OFFLINE_PROOF.equals(intent.getAction())) {
            AgentAccessibilityService svc = AgentAccessibilityService.getInstance();
            long delay = intent.getLongExtra("delay_ms", 15000);
            boolean armed = svc != null && svc.armOfflineSettingsProof(delay);
            setResultCode(armed ? 1 : 0);
            return;
        }
        if (ACTION_ARM_OFFLINE_IME_PROOF.equals(intent.getAction())) {
            AgentAccessibilityService svc = AgentAccessibilityService.getInstance();
            long delay = intent.getLongExtra("delay_ms", 15000);
            boolean armed = svc != null && svc.armOfflineImeProof(delay);
            setResultCode(armed ? 1 : 0);
            return;
        }
        if (ACTION_ARM_COMMUNICATION_PLANNER_PROOF.equals(intent.getAction())) {
            AgentAccessibilityService svc = AgentAccessibilityService.getInstance();
            long delay = intent.getLongExtra("delay_ms", 15000);
            String expectedSkill = intent.getStringExtra("expected_skill");
            boolean armed = svc != null
                    && svc.armCommunicationPlannerProof(expectedSkill, delay);
            setResultCode(armed ? 1 : 0);
            return;
        }
        if (!ACTION_PROBE.equals(intent.getAction())) {
            return;
        }
        try {
            JSONObject result = new JSONObject();
            result.put("captured_epoch_ms", System.currentTimeMillis());
            AgentAccessibilityService svc = AgentAccessibilityService.getInstance();
            result.put("accessibility_connected", svc != null);
            if (svc != null) {
                String targetHash = intent.getStringExtra("target_sha256");
                JSONObject summary = svc.snapshotSummary(targetHash);
                result.put("summary", summary);
                result.put("target_exact_semantic_match",
                        summary.optInt("target_hash_match_count", 0) > 0);
            }
            try (FileOutputStream out =
                         context.openFileOutput("semantic-probe.json", Context.MODE_PRIVATE)) {
                out.write(result.toString(2).getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception ignored) {
        }
    }
}
