package ai.zhiman.accessibility;

import android.content.Intent;
import android.content.SharedPreferences;
import android.net.Uri;

import org.json.JSONObject;

/**
 * Device-side executor for one already-planned and one-time-authorized communication skill.
 * Raw bindings stay in private SharedPreferences and are never included in proof files.
 */
public final class CommunicationSkillExecutor {
    public static final String PREF_PHONE = "bound_phone";
    public static final String PREF_WECHAT_NAME = "bound_wechat_display_name";
    public static final String PREF_MESSAGE = "approved_message_text";
    public static final String PREF_WECHAT_TITLE_FP = "wechat_title_fingerprint";

    private static final String WECHAT_PACKAGE = "com.tencent.mm";

    private CommunicationSkillExecutor() {
    }

    public static void execute(
            AgentAccessibilityService service,
            SharedPreferences bindings,
            PlannerClient.PlanResult plan,
            JSONObject proof) throws Exception {
        requireSupportedDisplay(service);
        if ("call_bound_contact".equals(plan.skill)) {
            executePhoneCall(service, bindings, proof);
        } else if ("send_wechat_message".equals(plan.skill)) {
            executeWechatMessage(service, bindings, proof);
        } else if ("start_wechat_video_call".equals(plan.skill)) {
            executeWechatVideoCall(service, bindings, proof);
        } else {
            throw new SecurityException("unsupported communication skill");
        }
    }

    public static String enrollBoundWechatTitle(
            AgentAccessibilityService service, SharedPreferences bindings)
            throws Exception {
        requireSupportedDisplay(service);
        String displayName = required(bindings, PREF_WECHAT_NAME);
        String apiKey = required(bindings, "mnn_api_key");
        Intent launch = service.getPackageManager().getLaunchIntentForPackage(WECHAT_PACKAGE);
        if (launch == null) {
            throw new IllegalStateException("WeChat is not installed");
        }
        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        service.startActivity(launch);
        sleep(1800);
        if (!service.tapNormalized(0.83f, 0.075f)) {
            throw new IllegalStateException("WeChat search tap failed during enrollment");
        }
        sleep(700);
        if (!service.tapNormalized(0.45f, 0.075f)) {
            throw new IllegalStateException("WeChat search field tap failed during enrollment");
        }
        sleep(500);
        AgentInputMethodService ime = AgentInputMethodService.getInstance();
        if (ime == null || !ime.commitTextNow(displayName)) {
            throw new IllegalStateException("headless IME could not enter bound search name");
        }
        sleep(1400);
        boolean unique = VisionVerifier.verifyUniqueContact(
                apiKey, service.captureScreenshotPng(10000), displayName);
        if (!unique) {
            throw new SecurityException("unique bound WeChat result was not verified");
        }
        if (!service.tapNormalized(0.50f, 0.19f)) {
            throw new IllegalStateException("WeChat result tap failed during enrollment");
        }
        sleep(1400);
        if (!WECHAT_PACKAGE.equals(service.getLastEventPackage())
                || !service.getLastEventClass().contains("ChattingUI")) {
            throw new SecurityException("WeChat direct-chat activity was not reached");
        }
        return VisualFingerprint.chatTitleSha256(service.captureScreenshotPng(10000));
    }

    private static void executePhoneCall(
            AgentAccessibilityService service,
            SharedPreferences bindings,
            JSONObject proof) throws Exception {
        String phone = required(bindings, PREF_PHONE);
        if (!phone.matches("[0-9]{3,20}")) {
            throw new SecurityException("invalid private phone binding");
        }
        Intent dial = new Intent(Intent.ACTION_DIAL, Uri.parse("tel:" + phone));
        dial.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        service.startActivity(dial);
        sleep(1800);

        JSONObject dialer = service.snapshotSummary(AgentAccessibilityService.sha256(phone));
        boolean numberVerified = dialer.optInt("target_hash_match_count", 0) == 1;
        proof.put("dialer_package", dialer.optString("package", ""));
        proof.put("bound_number_hash_verified", numberVerified);
        if (!numberVerified) {
            throw new SecurityException("dialer number does not match private binding");
        }

        boolean started = service.clickAnyViewId(
                "com.android.contacts:id/call_sim1",
                "com.android.contacts:id/call_button",
                "com.google.android.dialer:id/dialpad_floating_action_button");
        if (!started) {
            started = service.clickExactText("SIM 1") || service.clickExactText("拨号");
        }
        proof.put("call_button_clicked", started);
        if (!started) {
            throw new IllegalStateException("call button unavailable");
        }

        sleep(5000);
        boolean ended = service.clickAnyViewId(
                "com.android.incallui:id/endButton",
                "com.android.contacts:id/end_call_button",
                "com.google.android.dialer:id/incall_end_call");
        if (!ended) {
            ended = service.clickExactText("结束通话")
                    || service.clickExactText("挂断")
                    || service.clickNodeCenterByExactTextWithGestureFallback("结束通话");
        }
        proof.put("call_ended_by_agent", ended);
        if (!ended) {
            throw new IllegalStateException("could not verify call termination");
        }
    }

    private static void executeWechatMessage(
            AgentAccessibilityService service,
            SharedPreferences bindings,
            JSONObject proof) throws Exception {
        openBoundWechatChat(service, bindings, proof);
        String message = required(bindings, PREF_MESSAGE);
        if (message.length() > 200) {
            throw new SecurityException("approved message is too long");
        }
        if (!service.tapNormalized(0.43f, 0.945f)) {
            throw new IllegalStateException("message field tap failed");
        }
        sleep(600);
        AgentInputMethodService ime = AgentInputMethodService.getInstance();
        if (ime == null || !ime.commitTextNow(message)) {
            throw new IllegalStateException("headless IME could not commit approved message");
        }
        sleep(500);
        proof.put("approved_message_committed", true);
        if (!service.tapNormalized(0.91f, 0.945f)) {
            throw new IllegalStateException("send tap failed");
        }
        proof.put("send_tap_dispatched", true);
        sleep(1200);
    }

    private static void executeWechatVideoCall(
            AgentAccessibilityService service,
            SharedPreferences bindings,
            JSONObject proof) throws Exception {
        openBoundWechatChat(service, bindings, proof);
        if (!service.tapNormalized(0.94f, 0.945f)) {
            throw new IllegalStateException("WeChat action menu tap failed");
        }
        sleep(700);
        if (!service.tapNormalized(0.62f, 0.70f)) {
            throw new IllegalStateException("video-call tile tap failed");
        }
        sleep(700);
        if (!service.tapNormalized(0.50f, 0.83f)) {
            throw new IllegalStateException("video-call final option tap failed");
        }
        proof.put("video_call_start_tap_dispatched", true);
        sleep(5000);
        boolean ended = service.tapNormalized(0.50f, 0.86f);
        proof.put("video_call_end_tap_dispatched", ended);
        if (!ended) {
            throw new IllegalStateException("video-call hang-up tap failed");
        }
    }

    private static void openBoundWechatChat(
            AgentAccessibilityService service,
            SharedPreferences bindings,
            JSONObject proof) throws Exception {
        String displayName = required(bindings, PREF_WECHAT_NAME);
        String enrolledFingerprint = required(bindings, PREF_WECHAT_TITLE_FP);
        String apiKey = required(bindings, "mnn_api_key");

        Intent launch = service.getPackageManager().getLaunchIntentForPackage(WECHAT_PACKAGE);
        if (launch == null) {
            throw new IllegalStateException("WeChat is not installed");
        }
        launch.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TOP);
        service.startActivity(launch);
        sleep(1800);
        if (!service.tapNormalized(0.83f, 0.075f)) {
            throw new IllegalStateException("WeChat search tap failed");
        }
        sleep(700);
        if (!service.tapNormalized(0.45f, 0.075f)) {
            throw new IllegalStateException("WeChat search field tap failed");
        }
        sleep(500);
        AgentInputMethodService ime = AgentInputMethodService.getInstance();
        if (ime == null || !ime.commitTextNow(displayName)) {
            throw new IllegalStateException("headless IME could not enter bound search name");
        }
        sleep(1400);

        byte[] results = service.captureScreenshotPng(10000);
        boolean unique = VisionVerifier.verifyUniqueContact(apiKey, results, displayName);
        proof.put("unique_bound_search_result_verified", unique);
        if (!unique) {
            throw new SecurityException("unique bound WeChat result was not verified");
        }
        if (!service.tapNormalized(0.50f, 0.19f)) {
            throw new IllegalStateException("WeChat result tap failed");
        }
        sleep(1400);
        if (!WECHAT_PACKAGE.equals(service.getLastEventPackage())
                || !service.getLastEventClass().contains("ChattingUI")) {
            throw new SecurityException("WeChat direct-chat activity was not reached");
        }

        byte[] chat = service.captureScreenshotPng(10000);
        String currentFingerprint = VisualFingerprint.chatTitleSha256(chat);
        boolean strictFingerprintMatch =
                enrolledFingerprint.equalsIgnoreCase(currentFingerprint);
        proof.put("strict_enrolled_title_fingerprint_match", strictFingerprintMatch);
        boolean titleVisual = VisionVerifier.verifyChatTitle(apiKey, chat, displayName);
        proof.put("local_vision_title_verified", titleVisual);
        if (!strictFingerprintMatch || !titleVisual) {
            throw new SecurityException("active WeChat chat does not match private binding");
        }
        proof.put("bound_wechat_chat_verified", true);
    }

    private static String required(SharedPreferences prefs, String key) {
        String value = prefs.getString(key, "");
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalStateException("missing private binding: " + key);
        }
        return value.trim();
    }

    private static void requireSupportedDisplay(AgentAccessibilityService service) {
        int width = service.getResources().getDisplayMetrics().widthPixels;
        int height = service.getResources().getDisplayMetrics().heightPixels;
        if (width != 1220 || height != 2656) {
            throw new SecurityException("unvalidated display profile");
        }
    }

    private static void sleep(long millis) throws InterruptedException {
        Thread.sleep(millis);
    }
}
