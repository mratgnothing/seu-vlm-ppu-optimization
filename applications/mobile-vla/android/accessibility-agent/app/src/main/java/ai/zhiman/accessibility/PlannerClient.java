package ai.zhiman.accessibility;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.Iterator;

public final class PlannerClient {
    public static final class PlanResult {
        public final String skill;
        public final JSONObject arguments;
        public final long elapsedMs;
        public final String rawContent;

        PlanResult(String skill, JSONObject arguments, long elapsedMs, String rawContent) {
            this.skill = skill;
            this.arguments = arguments;
            this.elapsedMs = elapsedMs;
            this.rawContent = rawContent;
        }
    }

    private PlannerClient() {
    }

    public static PlanResult planOpenSettings(String apiKey) throws Exception {
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new IllegalStateException("MNN API key is not configured");
        }

        JSONObject payload = new JSONObject();
        payload.put("model", "Qwen3.5-2B-MNN");
        payload.put("temperature", 0);
        payload.put("max_tokens", 64);
        payload.put("enable_thinking", false);

        JSONArray messages = new JSONArray();
        messages.put(new JSONObject()
                .put("role", "system")
                .put("content",
                        "You are an offline phone skill router. " +
                        "The only permitted skill in this test is open_android_settings. " +
                        "Return exactly one JSON object with one key: " +
                        "{\"skill\":\"open_android_settings\"}. No prose."));
        messages.put(new JSONObject()
                .put("role", "user")
                .put("content", "Open Android Settings using the permitted low-risk local skill."));
        payload.put("messages", messages);

        URL url = new URL("http://127.0.0.1:8080/v1/chat/completions");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setConnectTimeout(8000);
        conn.setReadTimeout(120000);
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Authorization", "Bearer " + apiKey);
        conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");

        long start = System.currentTimeMillis();
        try (OutputStream os = conn.getOutputStream()) {
            os.write(payload.toString().getBytes(StandardCharsets.UTF_8));
        }

        int code = conn.getResponseCode();
        InputStream stream = code >= 200 && code < 300 ? conn.getInputStream() : conn.getErrorStream();
        String body = readAll(stream);
        long elapsed = System.currentTimeMillis() - start;
        if (code < 200 || code >= 300) {
            throw new IllegalStateException("MNN HTTP " + code);
        }

        JSONObject response = new JSONObject(body);
        String content = response
                .getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .optString("content", "")
                .trim();

        String jsonText = extractSingleJsonObject(content);
        JSONObject decision = new JSONObject(jsonText);
        Iterator<String> keys = decision.keys();
        int keyCount = 0;
        while (keys.hasNext()) {
            keys.next();
            keyCount++;
        }
        if (keyCount != 1) {
            throw new IllegalStateException("planner returned unexpected fields");
        }

        String skill = decision.optString("skill", "");
        if (!"open_android_settings".equals(skill)) {
            throw new IllegalStateException("planner chose a disallowed skill");
        }
        return new PlanResult(skill, new JSONObject(), elapsed, content);
    }

    public static PlanResult planCommunicationSkill(String apiKey, String userRequest)
            throws Exception {
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new IllegalStateException("MNN API key is not configured");
        }
        if (userRequest == null || userRequest.trim().isEmpty()) {
            throw new IllegalArgumentException("empty communication request");
        }

        JSONObject payload = new JSONObject();
        payload.put("model", "Qwen3.5-2B-MNN");
        payload.put("temperature", 0);
        payload.put("max_tokens", 128);
        payload.put("enable_thinking", false);
        payload.put("stream", false);

        String protocol =
                "You are an offline phone skill router. Return exactly one JSON object and no prose. " +
                "Choose exactly one of these schemas: " +
                "{\"skill\":\"call_bound_contact\",\"contact_alias\":\"authorized_test_contact\",\"max_duration_seconds\":5}; " +
                "{\"skill\":\"send_wechat_message\",\"contact_alias\":\"authorized_test_contact\",\"message_id\":\"skill_test_notice_v1\"}; " +
                "{\"skill\":\"start_wechat_video_call\",\"contact_alias\":\"authorized_test_contact\",\"max_ring_seconds\":5}. " +
                "Never invent a contact, number, message, duration, field, or second action.";

        JSONArray messages = new JSONArray();
        messages.put(new JSONObject().put("role", "system").put("content", protocol));
        messages.put(new JSONObject().put("role", "user").put("content", userRequest.trim()));
        payload.put("messages", messages);

        HttpResult result = postCompletion(apiKey, payload);
        String content = result.content;
        JSONObject decision = new JSONObject(extractSingleJsonObject(content));
        String skill = decision.optString("skill", "");
        validateCommunicationDecision(decision, skill);

        JSONObject arguments = new JSONObject();
        arguments.put("contact_alias", decision.getString("contact_alias"));
        if ("call_bound_contact".equals(skill)) {
            arguments.put("max_duration_seconds", decision.getInt("max_duration_seconds"));
        } else if ("send_wechat_message".equals(skill)) {
            arguments.put("message_id", decision.getString("message_id"));
        } else if ("start_wechat_video_call".equals(skill)) {
            arguments.put("max_ring_seconds", decision.getInt("max_ring_seconds"));
        }
        return new PlanResult(skill, arguments, result.elapsedMs, content);
    }

    public static void validateExpectedCommunicationPlan(
            PlanResult plan, String expectedSkill) {
        if (plan == null || !expectedSkill.equals(plan.skill)) {
            throw new SecurityException("planner skill does not match authorization");
        }
        if (!"authorized_test_contact".equals(
                plan.arguments.optString("contact_alias", ""))) {
            throw new SecurityException("contact alias is outside authorization");
        }
        if ("call_bound_contact".equals(expectedSkill)
                && plan.arguments.optInt("max_duration_seconds", -1) != 5) {
            throw new SecurityException("call duration is outside authorization");
        }
        if ("send_wechat_message".equals(expectedSkill)
                && !"skill_test_notice_v1".equals(
                plan.arguments.optString("message_id", ""))) {
            throw new SecurityException("message id is outside authorization");
        }
        if ("start_wechat_video_call".equals(expectedSkill)
                && plan.arguments.optInt("max_ring_seconds", -1) != 5) {
            throw new SecurityException("ring duration is outside authorization");
        }
    }

    public static String promptForExpectedCommunicationSkill(String expectedSkill) {
        if ("call_bound_contact".equals(expectedSkill)) {
            return "Call the already authorized contact and limit the call attempt to five seconds.";
        }
        if ("send_wechat_message".equals(expectedSkill)) {
            return "Send the pre-approved WeChat notice to the already authorized contact.";
        }
        if ("start_wechat_video_call".equals(expectedSkill)) {
            return "Start a WeChat video call to the already authorized contact and ring for at most five seconds.";
        }
        throw new IllegalArgumentException("unsupported expected skill");
    }

    private static void validateCommunicationDecision(JSONObject decision, String skill)
            throws Exception {
        int expectedFieldCount;
        if ("call_bound_contact".equals(skill)) {
            expectedFieldCount = 3;
            if (decision.optInt("max_duration_seconds", -1) != 5) {
                throw new SecurityException("invalid call duration");
            }
        } else if ("send_wechat_message".equals(skill)) {
            expectedFieldCount = 3;
            if (!"skill_test_notice_v1".equals(decision.optString("message_id", ""))) {
                throw new SecurityException("invalid message id");
            }
        } else if ("start_wechat_video_call".equals(skill)) {
            expectedFieldCount = 3;
            if (decision.optInt("max_ring_seconds", -1) != 5) {
                throw new SecurityException("invalid ring duration");
            }
        } else {
            throw new SecurityException("unknown communication skill");
        }
        if (!"authorized_test_contact".equals(decision.optString("contact_alias", ""))) {
            throw new SecurityException("invalid contact alias");
        }
        Iterator<String> keys = decision.keys();
        int keyCount = 0;
        while (keys.hasNext()) {
            keys.next();
            keyCount++;
        }
        if (keyCount != expectedFieldCount) {
            throw new SecurityException("planner returned unexpected fields");
        }
    }

    private static final class HttpResult {
        final String content;
        final long elapsedMs;

        HttpResult(String content, long elapsedMs) {
            this.content = content;
            this.elapsedMs = elapsedMs;
        }
    }

    private static HttpResult postCompletion(String apiKey, JSONObject payload) throws Exception {
        URL url = new URL("http://127.0.0.1:8080/v1/chat/completions");
        HttpURLConnection conn = (HttpURLConnection) url.openConnection();
        conn.setConnectTimeout(8000);
        conn.setReadTimeout(120000);
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Authorization", "Bearer " + apiKey);
        conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");

        long start = System.currentTimeMillis();
        try (OutputStream os = conn.getOutputStream()) {
            os.write(payload.toString().getBytes(StandardCharsets.UTF_8));
        }
        int code = conn.getResponseCode();
        InputStream stream = code >= 200 && code < 300 ? conn.getInputStream() : conn.getErrorStream();
        String body = readAll(stream);
        long elapsed = System.currentTimeMillis() - start;
        if (code < 200 || code >= 300) {
            throw new IllegalStateException("MNN HTTP " + code);
        }
        String content = new JSONObject(body)
                .getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .optString("content", "")
                .trim();
        return new HttpResult(content, elapsed);
    }

    private static String extractSingleJsonObject(String content) {
        int start = content.indexOf('{');
        int end = content.lastIndexOf('}');
        if (start < 0 || end <= start) {
            throw new IllegalArgumentException("planner did not return JSON");
        }
        return content.substring(start, end + 1);
    }

    private static String readAll(InputStream stream) throws Exception {
        if (stream == null) return "";
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                out.append(line);
            }
        }
        return out.toString();
    }
}
