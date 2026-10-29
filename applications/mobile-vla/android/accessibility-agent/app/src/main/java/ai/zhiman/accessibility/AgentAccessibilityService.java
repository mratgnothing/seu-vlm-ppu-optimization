package ai.zhiman.accessibility;

import android.accessibilityservice.AccessibilityService;
import android.accessibilityservice.AccessibilityServiceInfo;
import android.accessibilityservice.GestureDescription;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Bitmap;
import android.graphics.Path;
import android.graphics.Rect;
import android.hardware.HardwareBuffer;
import android.os.Bundle;
import android.provider.Settings;
import android.util.Log;
import android.view.Display;
import android.view.accessibility.AccessibilityEvent;
import android.view.accessibility.AccessibilityNodeInfo;

import org.json.JSONException;
import org.json.JSONArray;
import org.json.JSONObject;

import java.io.ByteArrayOutputStream;
import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.ArrayDeque;
import java.util.Locale;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.TimeUnit;

public final class AgentAccessibilityService extends AccessibilityService {
    private static final String TAG = "ZhiBanA11y";
    private static final String PREFS = "agent_config";
    private static final String PREF_API_KEY = "mnn_api_key";
    private static final String PROOF_FILE = "offline-proof.json";
    private static final String IME_PROOF_FILE = "offline-ime-proof.json";
    private static final String COMMUNICATION_PLANNER_PROOF_FILE =
            "communication-planner-proof.json";
    private static final String COMMUNICATION_EXECUTION_PROOF_FILE =
            "communication-execution-proof.json";
    private static final String WECHAT_ENROLLMENT_PROOF_FILE =
            "wechat-binding-enrollment.json";
    private static final String VISUAL_LOOP_PROOF_FILE = "visual-loop-proof.json";
    private static final String IME_PROBE_TEXT = "端侧输入测试";
    private static volatile AgentAccessibilityService instance;
    private volatile String lastEventPackage = "";
    private volatile String lastEventClass = "";
    private volatile boolean communicationBusy = false;
    private volatile boolean visualTaskBusy = false;
    private volatile JSONObject visualTaskProof;

    public static AgentAccessibilityService getInstance() {
        return instance;
    }

    @Override
    protected void onServiceConnected() {
        super.onServiceConnected();
        instance = this;
        AccessibilityServiceInfo info = getServiceInfo();
        info.flags |= AccessibilityServiceInfo.FLAG_REPORT_VIEW_IDS;
        info.flags |= AccessibilityServiceInfo.FLAG_INCLUDE_NOT_IMPORTANT_VIEWS;
        info.flags |= AccessibilityServiceInfo.FLAG_RETRIEVE_INTERACTIVE_WINDOWS;
        info.eventTypes = AccessibilityEvent.TYPES_ALL_MASK;
        info.feedbackType = AccessibilityServiceInfo.FEEDBACK_GENERIC;
        info.notificationTimeout = 50;
        setServiceInfo(info);
    }

    @Override
    public void onAccessibilityEvent(AccessibilityEvent event) {
        // Intentionally do not log node text or content descriptions.
        if (event != null) {
            lastEventPackage = safe(event.getPackageName());
            lastEventClass = safe(event.getClassName());
        }
    }

    public String getLastEventPackage() {
        return lastEventPackage;
    }

    public String getLastEventClass() {
        return lastEventClass;
    }

    /**
     * Returns true only when the current accessibility window belongs to WeChat.
     * Some HyperOS builds do not deliver a fresh window-state event after a
     * shell-level tap, so the active root is used as a secondary check.
     */
    public boolean isWechatWindowActive() {
        if ("com.tencent.mm".equals(lastEventPackage)
                && lastEventClass.contains("ChattingUI")) {
            return true;
        }
        try {
            AccessibilityNodeInfo root = getRootInActiveWindow();
            return root != null && "com.tencent.mm".contentEquals(root.getPackageName());
        } catch (Throwable ignored) {
            return false;
        }
    }

    @Override
    public void onInterrupt() {
    }

    @Override
    public void onDestroy() {
        if (instance == this) {
            instance = null;
        }
        super.onDestroy();
    }

    /**
     * Runs a bounded screenshot -> local VLM -> coordinate gate -> gesture loop.
     * The loop is intentionally restricted to this package's non-exported visual lab.
     */
    public synchronized boolean armSafeVisualTask(long delayMs) {
        if (visualTaskBusy) return false;
        long boundedDelay = Math.max(800, Math.min(delayMs, 10000));
        long armedAt = System.currentTimeMillis();
        visualTaskBusy = true;
        visualTaskProof = null;
        try {
            JSONObject armed = new JSONObject();
            armed.put("result", "ARMED");
            armed.put("armed_epoch_ms", armedAt);
            armed.put("scheduled_delay_ms", boundedDelay);
            armed.put("runner", "accessibility_service");
            armed.put("planner_location", "device_loopback_mnn");
            armed.put("allowed_package", getPackageName());
            armed.put("raw_screenshot_persisted", false);
            writeJsonFile(VISUAL_LOOP_PROOF_FILE, armed);
        } catch (Exception e) {
            visualTaskBusy = false;
            Log.e(TAG, "failed to arm visual task", e);
            return false;
        }

        new Thread(() -> {
            try {
                Thread.sleep(boundedDelay);
                runSafeVisualTask(armedAt);
            } catch (Throwable t) {
                Log.e(TAG, "safe visual task failed", t);
                try {
                    JSONObject fail = visualTaskProof == null
                            ? new JSONObject()
                            : visualTaskProof;
                    fail.put("result", "FAIL_CLOSED");
                    fail.put("armed_epoch_ms", armedAt);
                    fail.put("failed_epoch_ms", System.currentTimeMillis());
                    fail.put("error_type", t.getClass().getSimpleName());
                    fail.put("error_message", String.valueOf(t.getMessage()));
                    fail.put("raw_screenshot_persisted", false);
                    writeJsonFile(VISUAL_LOOP_PROOF_FILE, fail);
                } catch (Exception ignored) {
                }
            } finally {
                visualTaskBusy = false;
            }
        }, "a11y-safe-visual-loop").start();
        return true;
    }

    private void runSafeVisualTask(long armedAt) throws Exception {
        final int maximumSteps = 3;
        final String[] taskGoals = new String[]{
                "BLUE", "GREEN", "ORANGE"
        };
        JSONObject proof = new JSONObject();
        JSONArray stepEvidence = new JSONArray();
        proof.put("result", "RUNNING");
        proof.put("armed_epoch_ms", armedAt);
        proof.put("run_start_epoch_ms", System.currentTimeMillis());
        proof.put("runner", "accessibility_service");
        proof.put("planner_location", "device_loopback_mnn");
        proof.put("allowed_package", getPackageName());
        proof.put("maximum_steps", maximumSteps);
        proof.put("visual_action_space", "four_screen_quadrants");
        proof.put("raw_screenshot_persisted", false);
        proof.put("steps", stepEvidence);
        visualTaskProof = proof;
        writeJsonFile(VISUAL_LOOP_PROOF_FILE, proof);

        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        String apiKey = prefs.getString(PREF_API_KEY, "");
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new IllegalStateException("MNN API key is not configured");
        }

        for (int step = 1; step <= maximumSteps; step++) {
            String expectedMarker = "visual-task-target-step-" + step;
            if (!waitForVisualMarker(expectedMarker, 5000)) {
                throw new SecurityException("controlled visual target is not active");
            }
            if (!getPackageName().equals(currentRootPackage())) {
                throw new SecurityException("foreground package left the allowlist");
            }

            byte[] screenshot = captureScreenshotPng(10000);
            VisualActionPlanner.StepPlan plan =
                    VisualActionPlanner.planTap(
                            apiKey, screenshot, taskGoals[step - 1], step, maximumSteps);
            Rect target = uniqueTargetBounds();
            if (target == null || target.width() <= 0 || target.height() <= 0) {
                throw new SecurityException("unique visual target is unavailable");
            }

            int width = getResources().getDisplayMetrics().widthPixels;
            int height = getResources().getDisplayMetrics().heightPixels;
            int expectedChoice = target.centerY() < height / 2
                    ? (target.centerX() < width / 2 ? 1 : 2)
                    : (target.centerX() < width / 2 ? 3 : 4);
            int x = target.centerX();
            int y = target.centerY();
            boolean foregroundAllowed = getPackageName().equals(currentRootPackage());
            boolean insideScreen = x >= 0 && x < width && y >= 0 && y < height;
            boolean choiceMatchesTarget = plan.choice == expectedChoice;
            boolean insideTarget = insideScreen && target.contains(x, y);

            JSONObject one = new JSONObject();
            one.put("step", step);
            one.put("task_goal", taskGoals[step - 1]);
            one.put("planner_elapsed_ms", plan.elapsedMs);
            one.put("planned_choice", plan.choice);
            one.put("planner_exact_format", plan.exactFormat);
            one.put("expected_choice", expectedChoice);
            one.put("resolved_x", x);
            one.put("resolved_y", y);
            one.put("screen_width", width);
            one.put("screen_height", height);
            one.put("target_left", target.left);
            one.put("target_top", target.top);
            one.put("target_right", target.right);
            one.put("target_bottom", target.bottom);
            one.put("foreground_allowed", foregroundAllowed);
            one.put("choice_matches_target", choiceMatchesTarget);
            one.put("coordinate_inside_target", insideTarget);
            one.put("gate_decision",
                    foregroundAllowed && choiceMatchesTarget && insideTarget ? "ACT" : "BLOCK");
            stepEvidence.put(one);
            writeJsonFile(VISUAL_LOOP_PROOF_FILE, proof);

            if (!foregroundAllowed || !choiceMatchesTarget || !insideTarget) {
                throw new SecurityException("model choice failed target-boundary gate");
            }
            boolean dispatched = tapNormalized(
                    (float) x / width,
                    (float) y / height);
            one.put("gesture_dispatched", dispatched);
            writeJsonFile(VISUAL_LOOP_PROOF_FILE, proof);
            if (!dispatched) {
                throw new IllegalStateException("visual gesture dispatch failed");
            }

            String nextMarker = step == maximumSteps
                    ? "visual-task-complete"
                    : "visual-task-target-step-" + (step + 1);
            boolean transitionVerified = waitForVisualMarker(nextMarker, 5000);
            one.put("transition_verified", transitionVerified);
            writeJsonFile(VISUAL_LOOP_PROOF_FILE, proof);
            if (!transitionVerified || containsTextOrDescription("visual-task-wrong-tap")) {
                throw new SecurityException("visual task did not reach the expected next state");
            }
        }

        boolean completed = containsTextOrDescription("visual-task-complete");
        proof.put("completion_verified", completed);
        if (!completed) {
            throw new IllegalStateException("visual task completion marker is missing");
        }
        proof.put("result", "PASS");
        proof.put("finished_epoch_ms", System.currentTimeMillis());
        writeJsonFile(VISUAL_LOOP_PROOF_FILE, proof);
    }

    public synchronized boolean armOfflineSettingsProof(long delayMs) {
        long boundedDelay = Math.max(5000, Math.min(delayMs, 60000));
        long armedAt = System.currentTimeMillis();
        try {
            JSONObject armed = new JSONObject();
            armed.put("result", "ARMED");
            armed.put("armed_epoch_ms", armedAt);
            armed.put("scheduled_delay_ms", boundedDelay);
            armed.put("runner", "accessibility_service");
            armed.put("service_pid", android.os.Process.myPid());
            writeProof(armed);
            Log.i(TAG, "offline proof armed delayMs=" + boundedDelay);
        } catch (Exception e) {
            Log.e(TAG, "failed to persist armed state", e);
            return false;
        }

        new Thread(() -> {
            try {
                Thread.sleep(boundedDelay);
                runOfflineSettingsProof(armedAt);
            } catch (Throwable t) {
                Log.e(TAG, "offline proof failed", t);
                try {
                    JSONObject fail = new JSONObject();
                    fail.put("result", "FAIL");
                    fail.put("armed_epoch_ms", armedAt);
                    fail.put("failed_epoch_ms", System.currentTimeMillis());
                    fail.put("runner", "accessibility_service");
                    fail.put("error_type", t.getClass().getSimpleName());
                    fail.put("error_message", String.valueOf(t.getMessage()));
                    writeProof(fail);
                } catch (Exception ignored) {
                }
            }
        }, "a11y-offline-proof").start();
        return true;
    }

    public synchronized boolean armOfflineImeProof(long delayMs) {
        long boundedDelay = Math.max(5000, Math.min(delayMs, 60000));
        long armedAt = System.currentTimeMillis();
        try {
            JSONObject armed = new JSONObject();
            armed.put("result", "ARMED");
            armed.put("armed_epoch_ms", armedAt);
            armed.put("scheduled_delay_ms", boundedDelay);
            armed.put("runner", "accessibility_service");
            writeJsonFile(IME_PROOF_FILE, armed);
        } catch (Exception e) {
            Log.e(TAG, "failed to persist IME armed state", e);
            return false;
        }
        new Thread(() -> {
            try {
                Thread.sleep(boundedDelay);
                runOfflineImeProof(armedAt);
            } catch (Throwable t) {
                Log.e(TAG, "offline IME proof failed", t);
                try {
                    JSONObject fail = new JSONObject();
                    fail.put("result", "FAIL");
                    fail.put("armed_epoch_ms", armedAt);
                    fail.put("failed_epoch_ms", System.currentTimeMillis());
                    fail.put("runner", "accessibility_service");
                    fail.put("error_type", t.getClass().getSimpleName());
                    fail.put("error_message", String.valueOf(t.getMessage()));
                    writeJsonFile(IME_PROOF_FILE, fail);
                } catch (Exception ignored) {
                }
            }
        }, "a11y-offline-ime-proof").start();
        return true;
    }

    public synchronized boolean armCommunicationPlannerProof(
            String expectedSkill, long delayMs) {
        try {
            PlannerClient.promptForExpectedCommunicationSkill(expectedSkill);
        } catch (IllegalArgumentException e) {
            return false;
        }
        long boundedDelay = Math.max(5000, Math.min(delayMs, 60000));
        long armedAt = System.currentTimeMillis();
        try {
            JSONObject armed = new JSONObject();
            armed.put("result", "ARMED");
            armed.put("armed_epoch_ms", armedAt);
            armed.put("scheduled_delay_ms", boundedDelay);
            armed.put("expected_skill", expectedSkill);
            armed.put("runner", "accessibility_service");
            armed.put("external_action_attempted", false);
            writeJsonFile(COMMUNICATION_PLANNER_PROOF_FILE, armed);
        } catch (Exception e) {
            return false;
        }
        new Thread(() -> {
            try {
                Thread.sleep(boundedDelay);
                runCommunicationPlannerProof(expectedSkill, armedAt);
            } catch (Throwable t) {
                Log.e(TAG, "communication planner proof failed", t);
                try {
                    JSONObject fail = new JSONObject();
                    fail.put("result", "FAIL");
                    fail.put("armed_epoch_ms", armedAt);
                    fail.put("finished_epoch_ms", System.currentTimeMillis());
                    fail.put("expected_skill", expectedSkill);
                    fail.put("runner", "accessibility_service");
                    fail.put("error_type", t.getClass().getSimpleName());
                    fail.put("error_message", String.valueOf(t.getMessage()));
                    fail.put("external_action_attempted", false);
                    writeJsonFile(COMMUNICATION_PLANNER_PROOF_FILE, fail);
                } catch (Exception ignored) {
                }
            }
        }, "a11y-communication-planner-proof").start();
        return true;
    }

    private void runCommunicationPlannerProof(String expectedSkill, long armedAt)
            throws Exception {
        JSONObject proof = new JSONObject();
        proof.put("result", "RUNNING");
        proof.put("armed_epoch_ms", armedAt);
        proof.put("run_start_epoch_ms", System.currentTimeMillis());
        proof.put("runner", "accessibility_service");
        proof.put("host_executor_required", false);
        proof.put("planner_location", "device_loopback_mnn");
        proof.put("expected_skill", expectedSkill);
        proof.put("model_received_raw_contact_data", false);
        proof.put("external_action_attempted", false);
        writeJsonFile(COMMUNICATION_PLANNER_PROOF_FILE, proof);

        String apiKey = getSharedPreferences(PREFS, MODE_PRIVATE)
                .getString(PREF_API_KEY, "");
        String prompt = PlannerClient.promptForExpectedCommunicationSkill(expectedSkill);
        PlannerClient.PlanResult plan =
                PlannerClient.planCommunicationSkill(apiKey, prompt);
        proof.put("planner_skill", plan.skill);
        proof.put("planner_elapsed_ms", plan.elapsedMs);
        proof.put("planner_arguments", plan.arguments);

        try {
            PlannerClient.validateExpectedCommunicationPlan(plan, expectedSkill);
            proof.put("gate_decision", "ACT_DRY_RUN_ONLY");
        } catch (RuntimeException t) {
            proof.put("gate_decision", "BLOCK");
            writeJsonFile(COMMUNICATION_PLANNER_PROOF_FILE, proof);
            throw t;
        }
        proof.put("result", "PASS");
        proof.put("finished_epoch_ms", System.currentTimeMillis());
        writeJsonFile(COMMUNICATION_PLANNER_PROOF_FILE, proof);
    }

    /** Called only by the visible Activity after the user accepts a one-time confirmation dialog. */
    public synchronized boolean armAuthorizedCommunication(
            String expectedSkill, long delayMs) {
        if (communicationBusy) {
            return false;
        }
        try {
            PlannerClient.promptForExpectedCommunicationSkill(expectedSkill);
            validatePrivateBindings(expectedSkill);
        } catch (RuntimeException e) {
            return false;
        }
        communicationBusy = true;
        long boundedDelay = Math.max(3000, Math.min(delayMs, 15000));
        long armedAt = System.currentTimeMillis();
        try {
            JSONObject armed = new JSONObject();
            armed.put("result", "ARMED_BY_VISIBLE_USER_CONFIRMATION");
            armed.put("armed_epoch_ms", armedAt);
            armed.put("scheduled_delay_ms", boundedDelay);
            armed.put("expected_skill", expectedSkill);
            armed.put("authorization_single_use", true);
            armed.put("contains_raw_contact_data", false);
            writeJsonFile(COMMUNICATION_EXECUTION_PROOF_FILE, armed);
        } catch (Exception e) {
            communicationBusy = false;
            return false;
        }
        new Thread(() -> {
            try {
                Thread.sleep(boundedDelay);
                runAuthorizedCommunication(expectedSkill, armedAt);
            } catch (Throwable t) {
                Log.e(TAG, "authorized communication failed", t);
                try {
                    JSONObject fail = new JSONObject();
                    fail.put("result", "FAIL_CLOSED");
                    fail.put("armed_epoch_ms", armedAt);
                    fail.put("finished_epoch_ms", System.currentTimeMillis());
                    fail.put("expected_skill", expectedSkill);
                    fail.put("error_type", t.getClass().getSimpleName());
                    fail.put("error_message", String.valueOf(t.getMessage()));
                    fail.put("contains_raw_contact_data", false);
                    fail.put("external_action_may_have_been_attempted", true);
                    writeJsonFile(COMMUNICATION_EXECUTION_PROOF_FILE, fail);
                } catch (Exception ignored) {
                }
            } finally {
                communicationBusy = false;
            }
        }, "a11y-authorized-communication").start();
        return true;
    }

    private void runAuthorizedCommunication(String expectedSkill, long armedAt)
            throws Exception {
        JSONObject proof = new JSONObject();
        proof.put("result", "RUNNING");
        proof.put("armed_epoch_ms", armedAt);
        proof.put("run_start_epoch_ms", System.currentTimeMillis());
        proof.put("runner", "accessibility_service");
        proof.put("host_executor_required", false);
        proof.put("planner_location", "device_loopback_mnn");
        proof.put("expected_skill", expectedSkill);
        proof.put("authorization_single_use", true);
        proof.put("model_received_raw_contact_data", false);
        proof.put("contains_raw_contact_data", false);
        writeJsonFile(COMMUNICATION_EXECUTION_PROOF_FILE, proof);

        SharedPreferences bindings = getSharedPreferences(PREFS, MODE_PRIVATE);
        String prompt = PlannerClient.promptForExpectedCommunicationSkill(expectedSkill);
        PlannerClient.PlanResult plan = PlannerClient.planCommunicationSkill(
                bindings.getString(PREF_API_KEY, ""), prompt);
        proof.put("planner_skill", plan.skill);
        proof.put("planner_elapsed_ms", plan.elapsedMs);
        proof.put("planner_arguments", plan.arguments);
        try {
            PlannerClient.validateExpectedCommunicationPlan(plan, expectedSkill);
            validatePrivateBindings(expectedSkill);
            proof.put("gate_decision", "ACT_ONCE");
        } catch (RuntimeException e) {
            proof.put("gate_decision", "BLOCK");
            writeJsonFile(COMMUNICATION_EXECUTION_PROOF_FILE, proof);
            throw e;
        }
        writeJsonFile(COMMUNICATION_EXECUTION_PROOF_FILE, proof);

        proof.put("external_action_attempted", true);
        CommunicationSkillExecutor.execute(this, bindings, plan, proof);
        proof.put("result", "PASS");
        proof.put("finished_epoch_ms", System.currentTimeMillis());
        writeJsonFile(COMMUNICATION_EXECUTION_PROOF_FILE, proof);
    }

    public synchronized boolean armWechatBindingEnrollment(long delayMs) {
        if (communicationBusy) {
            return false;
        }
        communicationBusy = true;
        long boundedDelay = Math.max(8000, Math.min(delayMs, 30000));
        long armedAt = System.currentTimeMillis();
        new Thread(() -> {
            try {
                Thread.sleep(boundedDelay);
                SharedPreferences bindings = getSharedPreferences(PREFS, MODE_PRIVATE);
                String fingerprint = CommunicationSkillExecutor.enrollBoundWechatTitle(
                        this, bindings);
                bindings.edit()
                        .putString(
                                CommunicationSkillExecutor.PREF_WECHAT_TITLE_FP,
                                fingerprint)
                        .apply();
                JSONObject proof = new JSONObject();
                proof.put("result", "PASS");
                proof.put("armed_epoch_ms", armedAt);
                proof.put("finished_epoch_ms", System.currentTimeMillis());
                proof.put("binding", "authorized_test_contact");
                proof.put("fingerprint_algorithm", "quantized_title_crop_sha256");
                proof.put("raw_screenshot_persisted", false);
                proof.put("contains_raw_contact_data", false);
                writeJsonFile(WECHAT_ENROLLMENT_PROOF_FILE, proof);
                performGlobalAction(GLOBAL_ACTION_HOME);
            } catch (Throwable t) {
                try {
                    JSONObject fail = new JSONObject();
                    fail.put("result", "FAIL_CLOSED");
                    fail.put("armed_epoch_ms", armedAt);
                    fail.put("finished_epoch_ms", System.currentTimeMillis());
                    fail.put("error_type", t.getClass().getSimpleName());
                    fail.put("error_message", String.valueOf(t.getMessage()));
                    fail.put("raw_screenshot_persisted", false);
                    writeJsonFile(WECHAT_ENROLLMENT_PROOF_FILE, fail);
                } catch (Exception ignored) {
                }
            } finally {
                communicationBusy = false;
            }
        }, "a11y-wechat-binding-enrollment").start();
        return true;
    }

    private void validatePrivateBindings(String expectedSkill) {
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        if (prefs.getString(PREF_API_KEY, "").trim().isEmpty()) {
            throw new IllegalStateException("MNN API key is missing");
        }
        if ("call_bound_contact".equals(expectedSkill)) {
            String phone = prefs.getString(CommunicationSkillExecutor.PREF_PHONE, "");
            if (phone == null || !phone.trim().matches("[0-9]{3,20}")) {
                throw new IllegalStateException("private phone binding is missing");
            }
            return;
        }
        String name = prefs.getString(CommunicationSkillExecutor.PREF_WECHAT_NAME, "");
        String fingerprint = prefs.getString(
                CommunicationSkillExecutor.PREF_WECHAT_TITLE_FP, "");
        if (name == null || name.trim().isEmpty()
                || fingerprint == null || !fingerprint.matches("[0-9a-fA-F]{64}")) {
            throw new IllegalStateException("private WeChat binding is incomplete");
        }
        if ("send_wechat_message".equals(expectedSkill)) {
            String message = prefs.getString(CommunicationSkillExecutor.PREF_MESSAGE, "");
            if (message == null || message.trim().isEmpty() || message.length() > 200) {
                throw new IllegalStateException("approved private message is missing");
            }
        }
    }

    private void runOfflineImeProof(long armedAt) throws Exception {
        JSONObject proof = new JSONObject();
        proof.put("result", "RUNNING");
        proof.put("armed_epoch_ms", armedAt);
        proof.put("run_start_epoch_ms", System.currentTimeMillis());
        proof.put("runner", "accessibility_service");
        proof.put("host_executor_required", false);

        String defaultIme = Settings.Secure.getString(
                getContentResolver(), Settings.Secure.DEFAULT_INPUT_METHOD);
        boolean imeSelected = defaultIme != null
                && defaultIme.contains("ai.zhiman.accessibility/.AgentInputMethodService");
        proof.put("ime_selected", imeSelected);
        if (!imeSelected) {
            throw new IllegalStateException("Headless IME is not selected");
        }

        Intent target = new Intent(this, MainActivity.class);
        target.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK
                | Intent.FLAG_ACTIVITY_CLEAR_TOP
                | Intent.FLAG_ACTIVITY_SINGLE_TOP);
        target.putExtra("focus_ime_probe", true);
        startActivity(target);
        Thread.sleep(1800);

        AgentInputMethodService ime = AgentInputMethodService.getInstance();
        proof.put("ime_connected", ime != null);
        if (ime == null) {
            throw new IllegalStateException("Headless IME is not connected");
        }
        boolean committed = ime.commitTextNow(IME_PROBE_TEXT);
        proof.put("commit_text", committed);
        if (!committed) {
            throw new IllegalStateException("InputConnection.commitText failed");
        }
        Thread.sleep(600);

        JSONObject summary = snapshotSummary(sha256(IME_PROBE_TEXT));
        proof.put("input_snapshot", summary);
        boolean verified = summary.optInt("target_hash_match_count", 0) == 1;
        proof.put("unicode_text_verified", verified);
        if (!verified) {
            throw new IllegalStateException("committed Unicode text was not verified");
        }

        boolean homeAction = performGlobalAction(GLOBAL_ACTION_HOME);
        proof.put("accessibility_home_action", homeAction);
        if (!homeAction) {
            throw new IllegalStateException("GLOBAL_ACTION_HOME failed");
        }
        proof.put("result", "PASS");
        proof.put("finished_epoch_ms", System.currentTimeMillis());
        writeJsonFile(IME_PROOF_FILE, proof);
    }

    private void runOfflineSettingsProof(long armedAt) throws Exception {
        JSONObject proof = new JSONObject();
        proof.put("result", "RUNNING");
        proof.put("armed_epoch_ms", armedAt);
        proof.put("run_start_epoch_ms", System.currentTimeMillis());
        proof.put("runner", "accessibility_service");
        proof.put("service_pid", android.os.Process.myPid());
        proof.put("host_executor_required", false);
        proof.put("planner_location", "device_loopback_mnn");
        writeProof(proof);

        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        String apiKey = prefs.getString(PREF_API_KEY, "");
        PlannerClient.PlanResult plan = PlannerClient.planOpenSettings(apiKey);
        proof.put("planner_skill", plan.skill);
        proof.put("planner_elapsed_ms", plan.elapsedMs);

        if (!"open_android_settings".equals(plan.skill)) {
            proof.put("gate_decision", "BLOCK");
            writeProof(proof);
            throw new SecurityException("skill not allowed");
        }
        proof.put("gate_decision", "ACT");
        writeProof(proof);

        Intent settings = new Intent(Settings.ACTION_SETTINGS);
        settings.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        startActivity(settings);
        Thread.sleep(1800);

        JSONObject summary = snapshotSummary(null);
        proof.put("settings_snapshot", summary);
        String pkg = summary.optString("package", "");
        boolean settingsVerified = pkg.startsWith("com.android.settings");
        proof.put("settings_verified", settingsVerified);
        if (!settingsVerified || summary.optInt("node_count", 0) <= 0) {
            writeProof(proof);
            throw new IllegalStateException("Accessibility could not verify Settings UI");
        }

        boolean homeAction = performGlobalAction(GLOBAL_ACTION_HOME);
        proof.put("accessibility_home_action", homeAction);
        if (!homeAction) {
            writeProof(proof);
            throw new IllegalStateException("GLOBAL_ACTION_HOME failed");
        }

        proof.put("result", "PASS");
        proof.put("finished_epoch_ms", System.currentTimeMillis());
        writeProof(proof);
        Log.i(TAG, "offline proof PASS");
    }

    private synchronized void writeProof(JSONObject value) throws Exception {
        writeJsonFile(PROOF_FILE, value);
    }

    private synchronized void writeJsonFile(String name, JSONObject value) throws Exception {
        try (FileOutputStream out = openFileOutput(name, Context.MODE_PRIVATE)) {
            out.write(value.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }

    public synchronized JSONObject snapshotSummary(String expectedTextSha256) throws JSONException {
        JSONObject out = new JSONObject();
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) {
            out.put("root_available", false);
            out.put("node_count", 0);
            out.put("text_node_count", 0);
            out.put("view_id_count", 0);
            out.put("clickable_count", 0);
            out.put("editable_count", 0);
            out.put("target_hash_match_count", 0);
            return out;
        }

        out.put("root_available", true);
        out.put("package", safe(root.getPackageName()));
        out.put("class", safe(root.getClassName()));

        int nodeCount = 0;
        int textCount = 0;
        int viewIdCount = 0;
        int clickableCount = 0;
        int editableCount = 0;
        int targetHashMatches = 0;

        ArrayDeque<AccessibilityNodeInfo> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty() && nodeCount < 1000) {
            AccessibilityNodeInfo node = queue.removeFirst();
            nodeCount++;

            CharSequence text = node.getText();
            CharSequence desc = node.getContentDescription();
            if (notBlank(text) || notBlank(desc)) {
                textCount++;
            }
            if (notBlank(node.getViewIdResourceName())) {
                viewIdCount++;
            }
            if (node.isClickable()) {
                clickableCount++;
            }
            if (node.isEditable()) {
                editableCount++;
            }

            if (expectedTextSha256 != null && !expectedTextSha256.isEmpty()) {
                if (matchesHash(text, expectedTextSha256) || matchesHash(desc, expectedTextSha256)) {
                    targetHashMatches++;
                }
            }

            for (int i = 0; i < node.getChildCount(); i++) {
                AccessibilityNodeInfo child = node.getChild(i);
                if (child != null) {
                    queue.addLast(child);
                }
            }
        }

        out.put("node_count", nodeCount);
        out.put("text_node_count", textCount);
        out.put("view_id_count", viewIdCount);
        out.put("clickable_count", clickableCount);
        out.put("editable_count", editableCount);
        out.put("target_hash_match_count", targetHashMatches);
        return out;
    }

    public synchronized boolean clickExactText(String exactText) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null || exactText == null) {
            return false;
        }
        ArrayDeque<AccessibilityNodeInfo> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            AccessibilityNodeInfo node = queue.removeFirst();
            String text = safe(node.getText());
            String desc = safe(node.getContentDescription());
            if (exactText.equals(text) || exactText.equals(desc)) {
                AccessibilityNodeInfo candidate = node;
                while (candidate != null) {
                    if (candidate.isClickable() && candidate.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                        return true;
                    }
                    candidate = candidate.getParent();
                }
            }
            for (int i = 0; i < node.getChildCount(); i++) {
                AccessibilityNodeInfo child = node.getChild(i);
                if (child != null) {
                    queue.addLast(child);
                }
            }
        }
        return false;
    }

    public synchronized boolean clickViewId(String viewId) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null || viewId == null) {
            return false;
        }
        for (AccessibilityNodeInfo node : root.findAccessibilityNodeInfosByViewId(viewId)) {
            AccessibilityNodeInfo candidate = node;
            while (candidate != null) {
                if (candidate.isClickable() && candidate.performAction(AccessibilityNodeInfo.ACTION_CLICK)) {
                    return true;
                }
                candidate = candidate.getParent();
            }
        }
        return false;
    }

    public synchronized boolean clickAnyViewId(String... viewIds) {
        if (viewIds == null) return false;
        for (String viewId : viewIds) {
            if (clickViewId(viewId)) return true;
        }
        return false;
    }

    public synchronized boolean setTextOnFirstEditable(String text) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) {
            return false;
        }
        ArrayDeque<AccessibilityNodeInfo> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            AccessibilityNodeInfo node = queue.removeFirst();
            if (node.isEditable()) {
                Bundle args = new Bundle();
                args.putCharSequence(
                        AccessibilityNodeInfo.ACTION_ARGUMENT_SET_TEXT_CHARSEQUENCE,
                        text == null ? "" : text
                );
                if (node.performAction(AccessibilityNodeInfo.ACTION_SET_TEXT, args)) {
                    return true;
                }
            }
            for (int i = 0; i < node.getChildCount(); i++) {
                AccessibilityNodeInfo child = node.getChild(i);
                if (child != null) {
                    queue.addLast(child);
                }
            }
        }
        return false;
    }

    public boolean tapNormalized(float xFraction, float yFraction) {
        if (xFraction < 0f || xFraction > 1f || yFraction < 0f || yFraction > 1f) {
            return false;
        }
        int width = getResources().getDisplayMetrics().widthPixels;
        int height = getResources().getDisplayMetrics().heightPixels;
        Path path = new Path();
        path.moveTo(width * xFraction, height * yFraction);
        GestureDescription gesture = new GestureDescription.Builder()
                .addStroke(new GestureDescription.StrokeDescription(path, 0, 60))
                .build();
        CountDownLatch latch = new CountDownLatch(1);
        boolean[] accepted = new boolean[]{false};
        boolean[] completed = new boolean[]{false};
        getMainExecutor().execute(() -> {
            accepted[0] = dispatchGesture(
                    gesture,
                    new GestureResultCallback() {
                        @Override
                        public void onCompleted(GestureDescription gestureDescription) {
                            completed[0] = true;
                            latch.countDown();
                        }

                        @Override
                        public void onCancelled(GestureDescription gestureDescription) {
                            latch.countDown();
                        }
                    },
                    null);
            if (!accepted[0]) {
                latch.countDown();
            }
        });
        try {
            return latch.await(3000, TimeUnit.MILLISECONDS)
                    && accepted[0]
                    && completed[0];
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            return false;
        }
    }

    public byte[] captureScreenshotPng(long timeoutMs) throws Exception {
        final CountDownLatch latch = new CountDownLatch(1);
        final byte[][] resultHolder = new byte[1][];
        final int[] errorHolder = new int[]{0};

        takeScreenshot(
                Display.DEFAULT_DISPLAY,
                getMainExecutor(),
                new TakeScreenshotCallback() {
                    @Override
                    public void onSuccess(ScreenshotResult screenshot) {
                        HardwareBuffer buffer = screenshot.getHardwareBuffer();
                        try {
                            Bitmap hardwareBitmap = Bitmap.wrapHardwareBuffer(
                                    buffer,
                                    screenshot.getColorSpace()
                            );
                            if (hardwareBitmap == null) {
                                errorHolder[0] = -1;
                                return;
                            }
                            Bitmap bitmap = hardwareBitmap.copy(Bitmap.Config.ARGB_8888, false);
                            ByteArrayOutputStream output = new ByteArrayOutputStream();
                            bitmap.compress(Bitmap.CompressFormat.PNG, 100, output);
                            resultHolder[0] = output.toByteArray();
                            bitmap.recycle();
                        } finally {
                            buffer.close();
                            latch.countDown();
                        }
                    }

                    @Override
                    public void onFailure(int errorCode) {
                        errorHolder[0] = errorCode;
                        latch.countDown();
                    }
                });

        long boundedTimeout = Math.max(1000, Math.min(timeoutMs, 15000));
        if (!latch.await(boundedTimeout, TimeUnit.MILLISECONDS)) {
            throw new IllegalStateException("screenshot timed out");
        }
        if (resultHolder[0] == null) {
            throw new IllegalStateException("screenshot failed code=" + errorHolder[0]);
        }
        return resultHolder[0];
    }

    public synchronized boolean clickNodeCenterByExactTextWithGestureFallback(String exactText) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null || exactText == null) {
            return false;
        }
        ArrayDeque<AccessibilityNodeInfo> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            AccessibilityNodeInfo node = queue.removeFirst();
            if (exactText.equals(safe(node.getText())) || exactText.equals(safe(node.getContentDescription()))) {
                Rect bounds = new Rect();
                node.getBoundsInScreen(bounds);
                if (bounds.width() <= 0 || bounds.height() <= 0) {
                    return false;
                }
                int width = getResources().getDisplayMetrics().widthPixels;
                int height = getResources().getDisplayMetrics().heightPixels;
                return tapNormalized(
                        (float) bounds.centerX() / width,
                        (float) bounds.centerY() / height);
            }
            for (int i = 0; i < node.getChildCount(); i++) {
                AccessibilityNodeInfo child = node.getChild(i);
                if (child != null) queue.addLast(child);
            }
        }
        return false;
    }

    private synchronized String currentRootPackage() {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        return root == null ? "" : safe(root.getPackageName());
    }

    private synchronized Rect uniqueTargetBounds() {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null) return null;
        java.util.List<AccessibilityNodeInfo> nodes = root.findAccessibilityNodeInfosByViewId(
                getPackageName() + ":id/visual_task_target");
        if (nodes == null || nodes.size() != 1) return null;
        Rect bounds = new Rect();
        nodes.get(0).getBoundsInScreen(bounds);
        return bounds;
    }

    private synchronized boolean containsTextOrDescription(String exactValue) {
        AccessibilityNodeInfo root = getRootInActiveWindow();
        if (root == null || exactValue == null) return false;
        ArrayDeque<AccessibilityNodeInfo> queue = new ArrayDeque<>();
        queue.add(root);
        while (!queue.isEmpty()) {
            AccessibilityNodeInfo node = queue.removeFirst();
            if (exactValue.equals(safe(node.getText()))
                    || exactValue.equals(safe(node.getContentDescription()))) {
                return true;
            }
            for (int i = 0; i < node.getChildCount(); i++) {
                AccessibilityNodeInfo child = node.getChild(i);
                if (child != null) queue.addLast(child);
            }
        }
        return false;
    }

    private boolean waitForVisualMarker(String marker, long timeoutMs)
            throws InterruptedException {
        long deadline = System.currentTimeMillis() + Math.max(500, timeoutMs);
        while (System.currentTimeMillis() < deadline) {
            if (containsTextOrDescription(marker)) return true;
            Thread.sleep(100);
        }
        return false;
    }

    private static boolean matchesHash(CharSequence value, String expected) {
        if (!notBlank(value)) return false;
        return sha256(normalize(value.toString())).equalsIgnoreCase(expected);
    }

    private static String normalize(String value) {
        return value == null ? "" : value.trim().replaceAll("\\s+", " ");
    }

    public static String sha256(String value) {
        try {
            MessageDigest digest = MessageDigest.getInstance("SHA-256");
            byte[] result = digest.digest(value.getBytes(StandardCharsets.UTF_8));
            StringBuilder sb = new StringBuilder(result.length * 2);
            for (byte b : result) {
                sb.append(String.format(Locale.ROOT, "%02x", b));
            }
            return sb.toString();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static boolean notBlank(CharSequence value) {
        return value != null && value.toString().trim().length() > 0;
    }

    private static String safe(CharSequence value) {
        return value == null ? "" : value.toString();
    }
}
