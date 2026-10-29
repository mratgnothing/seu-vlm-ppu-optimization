package ai.zhiman.accessibility;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;

import org.json.JSONObject;

import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

/** Narrow bridge for the Android shell UID; protected by android.permission.DUMP. */
public final class ShellBridgeReceiver extends BroadcastReceiver {
    public static final String ACTION_COMMIT_NAME =
            "ai.zhiman.accessibility.shell.COMMIT_BOUND_WECHAT_NAME";
    public static final String ACTION_COMMIT_MESSAGE =
            "ai.zhiman.accessibility.shell.COMMIT_APPROVED_MESSAGE";
    public static final String ACTION_COMMIT_DEMO_CALL_PROMPT =
            "ai.zhiman.accessibility.shell.COMMIT_DEMO_CALL_PROMPT";
    public static final String ACTION_COMMIT_DEMO_COMBO_PROMPT =
            "ai.zhiman.accessibility.shell.COMMIT_DEMO_COMBO_PROMPT";
    public static final String ACTION_COMMIT_DEMO_DINNER_MESSAGE =
            "ai.zhiman.accessibility.shell.COMMIT_DEMO_DINNER_MESSAGE";
    public static final String ACTION_OPEN_BOUND_DIALER =
            "ai.zhiman.accessibility.shell.OPEN_BOUND_DIALER";
    public static final String ACTION_CAPTURE_CHAT_BINDING =
            "ai.zhiman.accessibility.shell.CAPTURE_CHAT_BINDING";
    public static final String ACTION_VERIFY_CHAT_BINDING =
            "ai.zhiman.accessibility.shell.VERIFY_CHAT_BINDING";

    @Override
    public void onReceive(Context context, Intent intent) {
        if (intent == null) return;
        String action = intent.getAction();
        if (ACTION_COMMIT_NAME.equals(action) || ACTION_COMMIT_MESSAGE.equals(action)) {
            SharedPreferences prefs = context.getSharedPreferences("agent_config", Context.MODE_PRIVATE);
            String key = ACTION_COMMIT_NAME.equals(action)
                    ? CommunicationSkillExecutor.PREF_WECHAT_NAME
                    : CommunicationSkillExecutor.PREF_MESSAGE;
            String value = prefs.getString(key, "");
            AgentInputMethodService ime = AgentInputMethodService.getInstance();
            boolean ok = ime != null && value != null && !value.isEmpty()
                    && ime.commitTextNow(value);
            setResultCode(ok ? 1 : 0);
            writeResult(context, action, ok, "commit");
            return;
        }
        if (ACTION_COMMIT_DEMO_CALL_PROMPT.equals(action)
                || ACTION_COMMIT_DEMO_COMBO_PROMPT.equals(action)
                || ACTION_COMMIT_DEMO_DINNER_MESSAGE.equals(action)) {
            SharedPreferences prefs = context.getSharedPreferences(
                    "agent_config", Context.MODE_PRIVATE);
            String displayName = prefs.getString(
                    CommunicationSkillExecutor.PREF_WECHAT_NAME, "");
            String value = "";
            if (displayName != null && !displayName.trim().isEmpty()
                    && displayName.length() <= 80) {
                if (ACTION_COMMIT_DEMO_CALL_PROMPT.equals(action)) {
                    value = "请给" + displayName.trim() + "打电话。";
                } else if (ACTION_COMMIT_DEMO_COMBO_PROMPT.equals(action)) {
                    value = "请给" + displayName.trim()
                            + "发微信，问他要不要一起出去吃饭，然后给他打视频电话。";
                } else {
                    value = "要不要一起出去吃饭？";
                }
            }
            AgentInputMethodService ime = AgentInputMethodService.getInstance();
            boolean ok = ime != null && !value.isEmpty() && ime.commitTextNow(value);
            setResultCode(ok ? 1 : 0);
            writeResult(context, action, ok, "demo_commit");
            return;
        }
        if (ACTION_OPEN_BOUND_DIALER.equals(action)) {
            boolean ok = false;
            try {
                SharedPreferences prefs = context.getSharedPreferences(
                        "agent_config", Context.MODE_PRIVATE);
                String phone = prefs.getString(CommunicationSkillExecutor.PREF_PHONE, "");
                if (phone == null || !phone.matches("[0-9]{3,20}")) {
                    throw new SecurityException("invalid private phone binding");
                }
                Intent dial = new Intent(Intent.ACTION_DIAL, Uri.parse("tel:" + phone));
                dial.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
                context.startActivity(dial);
                ok = true;
            } catch (Throwable ignored) {
                ok = false;
            }
            setResultCode(ok ? 1 : 0);
            writeResult(context, action, ok, "open_bound_dialer");
            return;
        }
        if (ACTION_CAPTURE_CHAT_BINDING.equals(action)
                || ACTION_VERIFY_CHAT_BINDING.equals(action)) {
            PendingResult pending = goAsync();
            new Thread(() -> {
                boolean ok = false;
                String operation = ACTION_CAPTURE_CHAT_BINDING.equals(action)
                        ? "capture" : "verify";
                try {
                    AgentAccessibilityService service = AgentAccessibilityService.getInstance();
                    if (service == null || !service.isWechatWindowActive()) {
                        throw new IllegalStateException("direct WeChat chat is not active");
                    }
                    String current = VisualFingerprint.chatTitleSha256(
                            service.captureScreenshotPng(10000));
                    SharedPreferences prefs = context.getSharedPreferences(
                            "agent_config", Context.MODE_PRIVATE);
                    if (ACTION_CAPTURE_CHAT_BINDING.equals(action)) {
                        prefs.edit().putString(
                                CommunicationSkillExecutor.PREF_WECHAT_TITLE_FP,
                                current).apply();
                        ok = true;
                    } else {
                        String enrolled = prefs.getString(
                                CommunicationSkillExecutor.PREF_WECHAT_TITLE_FP, "");
                        ok = enrolled.length() == 64 && enrolled.equalsIgnoreCase(current);
                    }
                } catch (Throwable ignored) {
                    ok = false;
                }
                pending.setResultCode(ok ? 1 : 0);
                writeResult(context, action, ok, operation);
                pending.finish();
            }, "shell-bridge-capture").start();
        }
    }

    private static void writeResult(
            Context context, String action, boolean ok, String operation) {
        try {
            JSONObject value = new JSONObject();
            value.put("result", ok ? "PASS" : "FAIL_CLOSED");
            value.put("operation", operation);
            value.put("action", action);
            value.put("finished_epoch_ms", System.currentTimeMillis());
            value.put("contains_raw_contact_data", false);
            value.put("raw_screenshot_persisted", false);
            try (FileOutputStream out = context.openFileOutput(
                    "shell-bridge-result.json", Context.MODE_PRIVATE)) {
                out.write(value.toString(2).getBytes(StandardCharsets.UTF_8));
            }
        } catch (Exception ignored) {
        }
    }
}
