package ai.zhiman.accessibility;

import android.accessibilityservice.AccessibilityService;
import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.content.Intent;
import android.content.SharedPreferences;
import android.os.Bundle;
import android.provider.Settings;
import android.text.InputType;
import android.view.inputmethod.InputMethodManager;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

import org.json.JSONObject;

import java.io.FileOutputStream;
import java.nio.charset.StandardCharsets;

public final class MainActivity extends Activity {
    private static final String PREFS = "agent_config";
    private static final String PREF_API_KEY = "mnn_api_key";
    private TextView status;
    private EditText apiKeyInput;
    private EditText imeTestInput;
    private EditText phoneBindingInput;
    private EditText wechatBindingInput;
    private EditText approvedMessageInput;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        buildUi();
        consumeIntent(getIntent());
    }

    @Override
    protected void onNewIntent(Intent intent) {
        super.onNewIntent(intent);
        setIntent(intent);
        consumeIntent(intent);
    }

    private void buildUi() {
        ScrollView scroll = new ScrollView(this);
        LinearLayout root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        int p = dp(16);
        root.setPadding(p, p, p, p);
        scroll.addView(root);

        TextView title = new TextView(this);
        title.setText("ZhiBan Accessibility Agent");
        title.setTextSize(22);
        root.addView(title);

        TextView description = new TextView(this);
        description.setText(
                "Local Qwen planning + deterministic gate + Android AccessibilityService. " +
                "Communication actions are not exposed by this debug screen.");
        description.setPadding(0, dp(8), 0, dp(16));
        root.addView(description);

        imeTestInput = new EditText(this);
        imeTestInput.setHint("Headless IME Unicode test target");
        imeTestInput.setSingleLine(true);
        imeTestInput.setInputType(InputType.TYPE_CLASS_TEXT);
        root.addView(imeTestInput);

        TextView bindingTitle = new TextView(this);
        bindingTitle.setText("Private authorized contact binding (stored only in app data)");
        bindingTitle.setPadding(0, dp(16), 0, dp(4));
        root.addView(bindingTitle);

        phoneBindingInput = new EditText(this);
        phoneBindingInput.setHint("Authorized phone number");
        phoneBindingInput.setSingleLine(true);
        phoneBindingInput.setInputType(InputType.TYPE_CLASS_PHONE);
        root.addView(phoneBindingInput);

        wechatBindingInput = new EditText(this);
        wechatBindingInput.setHint("Exact authorized WeChat display name");
        wechatBindingInput.setSingleLine(true);
        root.addView(wechatBindingInput);

        approvedMessageInput = new EditText(this);
        approvedMessageInput.setHint("One approved message (max 200 characters)");
        approvedMessageInput.setSingleLine(false);
        approvedMessageInput.setMaxLines(3);
        root.addView(approvedMessageInput);

        Button saveBinding = new Button(this);
        saveBinding.setText("Save private contact binding");
        saveBinding.setOnClickListener(v -> savePrivateBinding());
        root.addView(saveBinding);

        Button enrollWechat = new Button(this);
        enrollWechat.setText("Enroll current WeChat chat title (15s)");
        enrollWechat.setOnClickListener(v -> startWechatEnrollment());
        root.addView(enrollWechat);

        apiKeyInput = new EditText(this);
        apiKeyInput.setHint("MNN local API key");
        apiKeyInput.setSingleLine(true);
        apiKeyInput.setInputType(InputType.TYPE_CLASS_TEXT | InputType.TYPE_TEXT_VARIATION_PASSWORD);
        root.addView(apiKeyInput);

        Button saveKey = new Button(this);
        saveKey.setText("Save local MNN API key");
        saveKey.setOnClickListener(v -> {
            String key = apiKeyInput.getText().toString().trim();
            if (!key.isEmpty()) {
                prefs().edit().putString(PREF_API_KEY, key).apply();
                apiKeyInput.setText("");
                setStatus("API key saved locally.");
            }
        });
        root.addView(saveKey);

        Button accessibility = new Button(this);
        accessibility.setText("Open Accessibility settings");
        accessibility.setOnClickListener(v ->
                startActivity(new Intent(Settings.ACTION_ACCESSIBILITY_SETTINGS)));
        root.addView(accessibility);

        Button keyboardSettings = new Button(this);
        keyboardSettings.setText("Open keyboard settings");
        keyboardSettings.setOnClickListener(v ->
                startActivity(new Intent(Settings.ACTION_INPUT_METHOD_SETTINGS)));
        root.addView(keyboardSettings);

        Button imePicker = new Button(this);
        imePicker.setText("Show input method picker");
        imePicker.setOnClickListener(v -> {
            InputMethodManager manager =
                    (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
            if (manager != null) manager.showInputMethodPicker();
        });
        root.addView(imePicker);

        Button runtimeStatus = new Button(this);
        runtimeStatus.setText("Refresh runtime status");
        runtimeStatus.setOnClickListener(v -> showRuntimeStatus());
        root.addView(runtimeStatus);

        Button launchMnn = new Button(this);
        launchMnn.setText("Open MNN Chat");
        launchMnn.setOnClickListener(v -> {
            Intent launch = getPackageManager().getLaunchIntentForPackage("com.alibaba.mnnllm.android");
            if (launch != null) startActivity(launch);
        });
        root.addView(launchMnn);

        Button snapshot = new Button(this);
        snapshot.setText("Show structured UI summary");
        snapshot.setOnClickListener(v -> {
            try {
                AgentAccessibilityService svc = AgentAccessibilityService.getInstance();
                if (svc == null) {
                    setStatus("Accessibility service is not connected.");
                    return;
                }
                setStatus(svc.snapshotSummary(null).toString(2));
            } catch (Exception e) {
                setStatus("Snapshot failed: " + e.getClass().getSimpleName());
            }
        });
        root.addView(snapshot);

        Button arm = new Button(this);
        arm.setText("Arm 15s offline Settings proof");
        arm.setOnClickListener(v -> armOfflineSettingsProof(15000));
        root.addView(arm);

        Button armIme = new Button(this);
        armIme.setText("Arm 15s offline Headless IME proof");
        armIme.setOnClickListener(v -> {
            AgentAccessibilityService svc = AgentAccessibilityService.getInstance();
            if (svc == null) {
                setStatus("Accessibility service is not connected.");
                return;
            }
            boolean armed = svc.armOfflineImeProof(15000);
            setStatus(armed
                    ? "Offline IME proof armed. Host may stop ADB now."
                    : "Offline IME proof could not be armed.");
        });
        root.addView(armIme);

        Button visualLoop = new Button(this);
        visualLoop.setText("Run safe 3-step visual clicking test");
        visualLoop.setOnClickListener(v -> {
            AgentAccessibilityService svc = AgentAccessibilityService.getInstance();
            if (svc == null) {
                setStatus("Accessibility service is not connected.");
                return;
            }
            boolean armed = svc.armSafeVisualTask(1500);
            if (!armed) {
                setStatus("Visual task is already running or could not be armed.");
                return;
            }
            setStatus("Visual task armed. Only the controlled local task is allowed.");
            startActivity(new Intent(this, VisualTaskActivity.class));
        });
        root.addView(visualLoop);

        TextView warning = new TextView(this);
        warning.setText(
                "The buttons below create one single-use external action. " +
                "Each shows a final confirmation dialog and cannot be triggered by the exported probe receiver.");
        warning.setPadding(0, dp(16), 0, dp(4));
        root.addView(warning);

        Button call = new Button(this);
        call.setText("Authorize ONE 5s phone call");
        call.setOnClickListener(v -> confirmOneCommunication("call_bound_contact"));
        root.addView(call);

        Button message = new Button(this);
        message.setText("Authorize ONE approved WeChat message");
        message.setOnClickListener(v -> confirmOneCommunication("send_wechat_message"));
        root.addView(message);

        Button video = new Button(this);
        video.setText("Authorize ONE 5s WeChat video call");
        video.setOnClickListener(v -> confirmOneCommunication("start_wechat_video_call"));
        root.addView(video);

        status = new TextView(this);
        status.setText("Ready.");
        status.setTextIsSelectable(true);
        status.setPadding(0, dp(16), 0, dp(16));
        root.addView(status);

        setContentView(scroll);
    }

    private void consumeIntent(Intent intent) {
        if (intent == null) return;
        String key = intent.getStringExtra("mnn_api_key");
        if (key != null && !key.trim().isEmpty()) {
            prefs().edit().putString(PREF_API_KEY, key.trim()).apply();
            intent.removeExtra("mnn_api_key");
            setStatus("MNN key provisioned.");
        }
        if (intent.getBooleanExtra("arm_offline_settings_test", false)) {
            long delay = intent.getLongExtra("arm_delay_ms", 15000);
            intent.removeExtra("arm_offline_settings_test");
            armOfflineSettingsProof(Math.max(5000, Math.min(delay, 60000)));
        }
        if (intent.getBooleanExtra("focus_ime_probe", false)) {
            intent.removeExtra("focus_ime_probe");
            imeTestInput.setText("");
            imeTestInput.postDelayed(() -> {
                imeTestInput.requestFocus();
                imeTestInput.setSelection(imeTestInput.length());
                InputMethodManager manager =
                        (InputMethodManager) getSystemService(Context.INPUT_METHOD_SERVICE);
                if (manager != null) {
                    manager.showSoftInput(imeTestInput, InputMethodManager.SHOW_IMPLICIT);
                }
            }, 250);
        }
    }

    private void armOfflineSettingsProof(long delayMs) {
        final long armedAt = System.currentTimeMillis();
        setStatus("Offline proof armed for " + delayMs + " ms. Host may stop ADB now.");
        new Thread(() -> {
            try {
                Thread.sleep(delayMs);
                runOfflineSettingsProof(armedAt);
            } catch (Throwable t) {
                try {
                    JSONObject fail = new JSONObject();
                    fail.put("result", "FAIL");
                    fail.put("error_type", t.getClass().getSimpleName());
                    fail.put("finished_epoch_ms", System.currentTimeMillis());
                    writeJson("offline-proof.json", fail);
                } catch (Exception ignored) {
                }
            }
        }, "offline-proof").start();
    }

    private void runOfflineSettingsProof(long armedAt) throws Exception {
        JSONObject proof = new JSONObject();
        proof.put("armed_epoch_ms", armedAt);
        proof.put("run_start_epoch_ms", System.currentTimeMillis());
        proof.put("host_executor_required", false);
        proof.put("planner_location", "device_loopback_mnn");

        AgentAccessibilityService svc = AgentAccessibilityService.getInstance();
        if (svc == null) {
            throw new IllegalStateException("accessibility service is not connected");
        }
        proof.put("accessibility_connected", true);

        String apiKey = prefs().getString(PREF_API_KEY, "");
        PlannerClient.PlanResult plan = PlannerClient.planOpenSettings(apiKey);
        proof.put("planner_skill", plan.skill);
        proof.put("planner_elapsed_ms", plan.elapsedMs);

        if (!"open_android_settings".equals(plan.skill)) {
            proof.put("gate_decision", "BLOCK");
            throw new SecurityException("skill not allowed");
        }
        proof.put("gate_decision", "ACT");

        Intent settings = new Intent(Settings.ACTION_SETTINGS);
        settings.addFlags(Intent.FLAG_ACTIVITY_NEW_TASK);
        getApplicationContext().startActivity(settings);
        Thread.sleep(1800);

        JSONObject summary = svc.snapshotSummary(null);
        proof.put("settings_snapshot", summary);
        String pkg = summary.optString("package", "");
        boolean settingsVerified = pkg.startsWith("com.android.settings");
        proof.put("settings_verified", settingsVerified);
        if (!settingsVerified || summary.optInt("node_count", 0) <= 0) {
            throw new IllegalStateException("Accessibility could not verify Settings UI");
        }

        boolean homeAction = svc.performGlobalAction(AccessibilityService.GLOBAL_ACTION_HOME);
        proof.put("accessibility_home_action", homeAction);
        if (!homeAction) {
            throw new IllegalStateException("GLOBAL_ACTION_HOME failed");
        }

        proof.put("result", "PASS");
        proof.put("finished_epoch_ms", System.currentTimeMillis());
        writeJson("offline-proof.json", proof);
    }

    private void showRuntimeStatus() {
        boolean accessibilityConnected = AgentAccessibilityService.getInstance() != null;
        boolean imeConnected = AgentInputMethodService.getInstance() != null;
        String defaultIme = Settings.Secure.getString(
                getContentResolver(),
                Settings.Secure.DEFAULT_INPUT_METHOD);
        boolean imeSelected = defaultIme != null
                && defaultIme.contains("ai.zhiman.accessibility/.AgentInputMethodService");
        boolean apiKeyConfigured = !prefs().getString(PREF_API_KEY, "").isEmpty();
        setStatus(
                "accessibility_connected=" + accessibilityConnected +
                "\nime_selected=" + imeSelected +
                "\nime_connected=" + imeConnected +
                "\nmnn_api_key_configured=" + apiKeyConfigured);
    }

    private void savePrivateBinding() {
        String phone = phoneBindingInput.getText().toString().trim();
        String wechat = wechatBindingInput.getText().toString().trim();
        String message = approvedMessageInput.getText().toString();
        if (!phone.isEmpty() && !phone.matches("[0-9]{3,20}")) {
            setStatus("Phone binding must contain only 3-20 digits.");
            return;
        }
        if (message.length() > 200) {
            setStatus("Approved message exceeds 200 characters.");
            return;
        }
        SharedPreferences.Editor editor = prefs().edit();
        if (!phone.isEmpty()) {
            editor.putString(CommunicationSkillExecutor.PREF_PHONE, phone);
        }
        if (!wechat.isEmpty()) {
            editor.putString(CommunicationSkillExecutor.PREF_WECHAT_NAME, wechat);
        }
        if (!message.trim().isEmpty()) {
            editor.putString(CommunicationSkillExecutor.PREF_MESSAGE, message);
        }
        editor.apply();
        phoneBindingInput.setText("");
        wechatBindingInput.setText("");
        approvedMessageInput.setText("");
        setStatus("Private binding saved. Raw values were cleared from the screen.");
    }

    private void startWechatEnrollment() {
        String boundName = prefs().getString(
                CommunicationSkillExecutor.PREF_WECHAT_NAME, "");
        AgentAccessibilityService svc = AgentAccessibilityService.getInstance();
        if (svc == null || boundName == null || boundName.trim().isEmpty()) {
            setStatus("Save a WeChat binding and enable Accessibility first.");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("Enroll private WeChat binding")
                .setMessage(
                        "After continuing, the Agent will search locally for “" +
                        boundName +
                        "”, verify one result, open the direct chat, and store only a quantized title-region SHA-256 fingerprint. No screenshot is saved and no message is sent.")
                .setNegativeButton("Cancel", null)
                .setPositiveButton("Continue", (dialog, which) -> {
                    boolean armed = svc.armWechatBindingEnrollment(15000);
                    if (!armed) {
                        setStatus("Enrollment could not be armed.");
                        return;
                    }
                    Intent launch = getPackageManager().getLaunchIntentForPackage("com.tencent.mm");
                    if (launch == null) {
                        setStatus("WeChat is not installed.");
                        return;
                    }
                    startActivity(launch);
                })
                .show();
    }

    private void confirmOneCommunication(String skill) {
        AgentAccessibilityService svc = AgentAccessibilityService.getInstance();
        if (svc == null) {
            setStatus("Accessibility service is not connected.");
            return;
        }
        String target;
        String action;
        if ("call_bound_contact".equals(skill)) {
            target = prefs().getString(CommunicationSkillExecutor.PREF_PHONE, "");
            action = "Place one phone call and end it after five seconds";
        } else if ("send_wechat_message".equals(skill)) {
            target = prefs().getString(CommunicationSkillExecutor.PREF_WECHAT_NAME, "");
            action = "Send exactly this approved message:\n" +
                    prefs().getString(CommunicationSkillExecutor.PREF_MESSAGE, "");
        } else {
            target = prefs().getString(CommunicationSkillExecutor.PREF_WECHAT_NAME, "");
            action = "Start one WeChat video call and end it after five seconds";
        }
        if (target == null || target.trim().isEmpty()) {
            setStatus("The required private target binding is missing.");
            return;
        }
        new AlertDialog.Builder(this)
                .setTitle("Single-use external action")
                .setMessage("Target: " + target + "\n\nAction: " + action +
                        "\n\nLocal Qwen will select one skill; the deterministic gate must match this exact authorization.")
                .setNegativeButton("Cancel", null)
                .setPositiveButton("Authorize once", (dialog, which) -> {
                    boolean armed = svc.armAuthorizedCommunication(skill, 5000);
                    setStatus(armed
                            ? "One-time action armed. Keep the phone unlocked."
                            : "Action blocked: runtime or private binding is incomplete.");
                })
                .show();
    }

    private SharedPreferences prefs() {
        return getSharedPreferences(PREFS, MODE_PRIVATE);
    }

    private void writeJson(String name, JSONObject value) throws Exception {
        try (FileOutputStream out = openFileOutput(name, Context.MODE_PRIVATE)) {
            out.write(value.toString(2).getBytes(StandardCharsets.UTF_8));
        }
    }

    private void setStatus(String text) {
        if (status == null) return;
        runOnUiThread(() -> status.setText(text));
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
