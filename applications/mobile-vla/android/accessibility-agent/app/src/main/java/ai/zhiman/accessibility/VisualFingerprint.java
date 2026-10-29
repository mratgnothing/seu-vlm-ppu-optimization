package ai.zhiman.accessibility;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.Color;

import java.security.MessageDigest;
import java.util.Locale;

/** A strict, quantized SHA-256 fingerprint for the stable center of a WeChat title bar. */
public final class VisualFingerprint {
    private VisualFingerprint() {
    }

    public static String chatTitleSha256(byte[] screenshotPng) {
        Bitmap source = BitmapFactory.decodeByteArray(
                screenshotPng, 0, screenshotPng.length);
        if (source == null) {
            throw new IllegalArgumentException("invalid screenshot");
        }
        int left = Math.round(source.getWidth() * 0.20f);
        int top = Math.round(source.getHeight() * 0.045f);
        int width = Math.max(9, Math.round(source.getWidth() * 0.60f));
        int height = Math.max(8, Math.round(source.getHeight() * 0.075f));
        width = Math.min(width, source.getWidth() - left);
        height = Math.min(height, source.getHeight() - top);
        Bitmap crop = Bitmap.createBitmap(source, left, top, width, height);
        Bitmap small = Bitmap.createScaledBitmap(crop, 128, 32, true);
        byte[] quantized = new byte[128 * 32];
        int index = 0;
        for (int y = 0; y < 32; y++) {
            for (int x = 0; x < 128; x++) {
                quantized[index++] = (byte) (luma(small.getPixel(x, y)) >>> 4);
            }
        }
        small.recycle();
        crop.recycle();
        source.recycle();
        try {
            byte[] digest = MessageDigest.getInstance("SHA-256").digest(quantized);
            StringBuilder out = new StringBuilder(64);
            for (byte value : digest) {
                out.append(String.format(Locale.ROOT, "%02x", value));
            }
            return out.toString();
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }
    }

    private static int luma(int color) {
        return (299 * Color.red(color) + 587 * Color.green(color)
                + 114 * Color.blue(color)) / 1000;
    }
}
