package ai.zhiman.accessibility;

import android.util.Base64;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
/** Strict screenshot-to-discrete-target planner for the controlled visual task. */
public final class VisualActionPlanner {
    public static final class StepPlan {
        public final int choice;
        public final long elapsedMs;
        public final boolean exactFormat;

        StepPlan(int choice, long elapsedMs, boolean exactFormat) {
            this.choice = choice;
            this.elapsedMs = elapsedMs;
            this.exactFormat = exactFormat;
        }
    }

    private VisualActionPlanner() {
    }

    public static StepPlan planTap(
            String apiKey,
            byte[] screenshotPng,
            String taskGoal,
            int stepNumber,
            int maximumSteps) throws Exception {
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new IllegalStateException("MNN API key is not configured");
        }
        if (screenshotPng == null || screenshotPng.length == 0) {
            throw new IllegalArgumentException("empty screenshot");
        }
        if (stepNumber < 1 || stepNumber > maximumSteps || maximumSteps > 5) {
            throw new IllegalArgumentException("invalid visual step");
        }
        if (taskGoal == null || !taskGoal.matches("[A-Z]+")) {
            throw new IllegalArgumentException("invalid controlled task goal");
        }

        String prompt =
                "You are the visual target selector for a controlled local Android test. " +
                "User goal: tap the large card labeled exactly " + taskGoal + ". " +
                "Look at the screenshot. Each of the four large cards visibly begins with " +
                "its number 1, 2, 3, or 4 above the label. " +
                "Return the visible number printed on the card with the exact requested label. " +
                "This is step " + stepNumber + " of " + maximumSteps + ". " +
                "Reply with exactly one digit: 1, 2, 3, or 4. No words, JSON, punctuation, or markdown.";

        String encoded = Base64.encodeToString(screenshotPng, Base64.NO_WRAP);
        JSONArray content = new JSONArray();
        content.put(new JSONObject().put("type", "text").put("text", prompt));
        content.put(new JSONObject()
                .put("type", "image_url")
                .put("image_url", new JSONObject()
                        .put("url", "data:image/png;base64," + encoded)));

        JSONObject payload = new JSONObject();
        payload.put("model", "Qwen3.5-2B-MNN");
        payload.put("stream", false);
        payload.put("temperature", 0);
        payload.put("max_tokens", 4);
        payload.put("enable_thinking", false);
        payload.put("messages", new JSONArray()
                .put(new JSONObject().put("role", "user").put("content", content)));

        HttpURLConnection connection = (HttpURLConnection)
                new URL("http://127.0.0.1:8080/v1/chat/completions").openConnection();
        connection.setConnectTimeout(8000);
        connection.setReadTimeout(120000);
        connection.setRequestMethod("POST");
        connection.setDoOutput(true);
        connection.setRequestProperty("Authorization", "Bearer " + apiKey);
        connection.setRequestProperty("Content-Type", "application/json; charset=utf-8");

        long started = System.currentTimeMillis();
        try (OutputStream output = connection.getOutputStream()) {
            output.write(payload.toString().getBytes(StandardCharsets.UTF_8));
        }
        int status = connection.getResponseCode();
        InputStream stream = status >= 200 && status < 300
                ? connection.getInputStream()
                : connection.getErrorStream();
        String body = readAll(stream);
        long elapsed = System.currentTimeMillis() - started;
        if (status < 200 || status >= 300) {
            throw new IllegalStateException("MNN visual planner HTTP " + status);
        }

        String answer = new JSONObject(body)
                .getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .optString("content", "")
                .trim();
        if (answer.matches("[1-4]")) {
            return new StepPlan(Integer.parseInt(answer), elapsed, true);
        }
        if (answer.length() > 4096) {
            throw new SecurityException("visual planner response is too long");
        }
        String terminal = answer.replaceAll("[\\s`]+$", "");
        Matcher terminalChoice = Pattern.compile("(?<![0-9])([1-4])$").matcher(terminal);
        if (terminalChoice.find()) {
            return new StepPlan(
                    Integer.parseInt(terminalChoice.group(1)), elapsed, false);
        }
        Matcher choices = Pattern.compile("(?<![0-9])([1-4])(?![0-9])").matcher(answer);
        int uniqueChoice = -1;
        while (choices.find()) {
            int candidate = Integer.parseInt(choices.group(1));
            if (uniqueChoice == -1) {
                uniqueChoice = candidate;
            } else if (uniqueChoice != candidate) {
                throw new SecurityException("visual planner returned conflicting choices");
            }
        }
        if (uniqueChoice == -1) {
            throw new SecurityException("visual planner returned no discrete action");
        }
        return new StepPlan(uniqueChoice, elapsed, false);
    }

    private static String readAll(InputStream stream) throws Exception {
        if (stream == null) return "";
        StringBuilder output = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) output.append(line);
        }
        return output.toString();
    }
}
