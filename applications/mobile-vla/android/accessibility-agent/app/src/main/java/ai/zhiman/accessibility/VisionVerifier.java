package ai.zhiman.accessibility;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Base64;

import org.json.JSONArray;
import org.json.JSONObject;

import java.io.BufferedReader;
import java.io.ByteArrayOutputStream;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.nio.charset.StandardCharsets;

public final class VisionVerifier {
    private VisionVerifier() {
    }

    public static boolean verifyUniqueContact(
            String apiKey,
            byte[] screenshotPng,
            String expectedDisplayName) throws Exception {
        byte[] focused = cropAndScaleFirstResult(screenshotPng);
        String prompt =
                "This is a cropped WeChat search-result region. " +
                "Is there exactly one visible contact result whose displayed name is exactly " +
                expectedDisplayName +
                "? Reply exactly YES or NO.";
        String answer = askVision(apiKey, focused, prompt).trim();
        return "YES".equalsIgnoreCase(answer);
    }

    public static boolean verifyChatTitle(
            String apiKey,
            byte[] screenshotPng,
            String expectedDisplayName) throws Exception {
        byte[] focused = cropAndScaleTopTitle(screenshotPng);
        String prompt =
                "This is the top-title region of a WeChat chat screen. " +
                "Does the displayed chat title exactly equal " + expectedDisplayName +
                "? Reply exactly YES or NO.";
        String answer = askVision(apiKey, focused, prompt).trim();
        return "YES".equalsIgnoreCase(answer);
    }

    private static String askVision(String apiKey, byte[] png, String prompt) throws Exception {
        if (apiKey == null || apiKey.trim().isEmpty()) {
            throw new IllegalStateException("MNN API key is not configured");
        }

        String b64 = Base64.encodeToString(png, Base64.NO_WRAP);
        JSONArray content = new JSONArray();
        content.put(new JSONObject().put("type", "text").put("text", prompt));
        content.put(new JSONObject()
                .put("type", "image_url")
                .put("image_url", new JSONObject()
                        .put("url", "data:image/png;base64," + b64)));

        JSONObject payload = new JSONObject();
        payload.put("model", "Qwen3.5-2B-MNN");
        payload.put("stream", false);
        payload.put("temperature", 0);
        payload.put("max_tokens", 16);
        payload.put("enable_thinking", false);
        payload.put("messages", new JSONArray()
                .put(new JSONObject().put("role", "user").put("content", content)));

        HttpURLConnection conn = (HttpURLConnection)
                new URL("http://127.0.0.1:8080/v1/chat/completions").openConnection();
        conn.setConnectTimeout(8000);
        conn.setReadTimeout(120000);
        conn.setRequestMethod("POST");
        conn.setDoOutput(true);
        conn.setRequestProperty("Authorization", "Bearer " + apiKey);
        conn.setRequestProperty("Content-Type", "application/json; charset=utf-8");
        try (OutputStream out = conn.getOutputStream()) {
            out.write(payload.toString().getBytes(StandardCharsets.UTF_8));
        }

        int code = conn.getResponseCode();
        InputStream stream = code >= 200 && code < 300
                ? conn.getInputStream()
                : conn.getErrorStream();
        String body = readAll(stream);
        if (code < 200 || code >= 300) {
            throw new IllegalStateException("MNN vision HTTP " + code);
        }

        return new JSONObject(body)
                .getJSONArray("choices")
                .getJSONObject(0)
                .getJSONObject("message")
                .optString("content", "");
    }

    private static byte[] cropAndScaleFirstResult(byte[] png) {
        Bitmap bitmap = BitmapFactory.decodeByteArray(png, 0, png.length);
        if (bitmap == null) throw new IllegalArgumentException("invalid screenshot");
        int left = 0;
        int top = Math.max(0, Math.round(bitmap.getHeight() * 0.10f));
        int width = bitmap.getWidth();
        int height = Math.min(
                bitmap.getHeight() - top,
                Math.max(1, Math.round(bitmap.getHeight() * 0.30f)));
        return cropScaleEncode(bitmap, left, top, width, height);
    }

    private static byte[] cropAndScaleTopTitle(byte[] png) {
        Bitmap bitmap = BitmapFactory.decodeByteArray(png, 0, png.length);
        if (bitmap == null) throw new IllegalArgumentException("invalid screenshot");
        int left = 0;
        int top = 0;
        int width = bitmap.getWidth();
        int height = Math.max(1, Math.round(bitmap.getHeight() * 0.14f));
        return cropScaleEncode(bitmap, left, top, width, height);
    }

    private static byte[] cropScaleEncode(
            Bitmap source, int left, int top, int width, int height) {
        Bitmap crop = Bitmap.createBitmap(source, left, top, width, height);
        int targetWidth = Math.min(2400, crop.getWidth() * 2);
        int targetHeight = Math.max(1,
                Math.round((float) crop.getHeight() * targetWidth / crop.getWidth()));
        Bitmap scaled = Bitmap.createScaledBitmap(crop, targetWidth, targetHeight, true);
        ByteArrayOutputStream output = new ByteArrayOutputStream();
        scaled.compress(Bitmap.CompressFormat.PNG, 100, output);
        if (crop != source) crop.recycle();
        scaled.recycle();
        source.recycle();
        return output.toByteArray();
    }

    private static String readAll(InputStream stream) throws Exception {
        if (stream == null) return "";
        StringBuilder out = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(stream, StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) out.append(line);
        }
        return out.toString();
    }
}
