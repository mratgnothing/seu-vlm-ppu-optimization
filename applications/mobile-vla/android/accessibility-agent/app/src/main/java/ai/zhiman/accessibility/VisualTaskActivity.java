package ai.zhiman.accessibility;

import android.app.Activity;
import android.graphics.Color;
import android.graphics.Typeface;
import android.os.Bundle;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.GridLayout;
import android.widget.LinearLayout;
import android.widget.TextView;

/** A local, side-effect-free screen used to measure multi-step visual pointing. */
public final class VisualTaskActivity extends Activity {
    private static final String[][] LABELS = new String[][]{
            {"RED", "GREEN", "ORANGE", "BLUE"},
            {"GREEN", "PURPLE", "BLUE", "ORANGE"},
            {"PURPLE", "BLUE", "ORANGE", "GREEN"}
    };
    private static final int[] TARGET_INDEX = new int[]{3, 0, 2};
    private static final int[] CARD_COLORS = new int[]{
            Color.rgb(198, 56, 56),
            Color.rgb(45, 150, 85),
            Color.rgb(232, 137, 42),
            Color.rgb(48, 105, 190),
            Color.rgb(132, 73, 168)
    };

    private LinearLayout root;
    private int step;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        step = 0;
        renderStep();
    }

    private void renderStep() {
        root = new LinearLayout(this);
        root.setOrientation(LinearLayout.VERTICAL);
        root.setGravity(Gravity.CENTER_HORIZONTAL);
        root.setPadding(dp(18), dp(12), dp(18), dp(12));
        root.setBackgroundColor(Color.rgb(246, 247, 250));

        TextView title = new TextView(this);
        title.setText("SAFE VISUAL TASK  " + (step + 1) + "/3");
        title.setTextColor(Color.rgb(22, 31, 45));
        title.setTextSize(24);
        title.setTypeface(Typeface.DEFAULT_BOLD);
        title.setGravity(Gravity.CENTER);
        root.addView(title, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(56)));

        int targetIndex = TARGET_INDEX[step];
        String targetLabel = LABELS[step][targetIndex];
        TextView instruction = new TextView(this);
        instruction.setText("TAP THE LARGE CARD:  " + targetLabel);
        instruction.setTextColor(Color.WHITE);
        instruction.setTextSize(21);
        instruction.setTypeface(Typeface.DEFAULT_BOLD);
        instruction.setGravity(Gravity.CENTER);
        instruction.setBackgroundColor(Color.rgb(35, 45, 62));
        root.addView(instruction, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(86)));

        TextView note = new TextView(this);
        note.setText("The local model sees this screen and must choose the numbered target.");
        note.setTextColor(Color.DKGRAY);
        note.setTextSize(15);
        note.setGravity(Gravity.CENTER);
        root.addView(note, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(55)));

        GridLayout grid = new GridLayout(this);
        grid.setColumnCount(2);
        grid.setRowCount(2);
        grid.setUseDefaultMargins(true);
        root.addView(grid, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));

        for (int i = 0; i < 4; i++) {
            Button card = new Button(this);
            card.setText((i + 1) + "\n" + LABELS[step][i]);
            card.setTextColor(Color.WHITE);
            card.setTextSize(19);
            card.setTypeface(Typeface.DEFAULT_BOLD);
            card.setGravity(Gravity.CENTER);
            card.setBackgroundColor(CARD_COLORS[colorIndex(LABELS[step][i])]);
            GridLayout.LayoutParams params = new GridLayout.LayoutParams();
            params.width = 0;
            params.height = 0;
            params.columnSpec = GridLayout.spec(i % 2, 1f);
            params.rowSpec = GridLayout.spec(i / 2, 1f);
            params.setMargins(dp(8), dp(8), dp(8), dp(8));
            card.setLayoutParams(params);
            if (i == targetIndex) {
                card.setId(R.id.visual_task_target);
                card.setContentDescription("visual-task-target-step-" + (step + 1));
                card.setOnClickListener(v -> advance());
            } else {
                card.setContentDescription("visual-task-distractor");
                card.setOnClickListener(v -> showBlockedWrongTap());
            }
            grid.addView(card);
        }

        TextView safety = new TextView(this);
        safety.setText("Controlled local test • no network target • no settings changed");
        safety.setTextColor(Color.GRAY);
        safety.setTextSize(14);
        safety.setGravity(Gravity.CENTER);
        root.addView(safety, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(48)));
        setContentView(root);
    }

    private void advance() {
        step++;
        if (step >= TARGET_INDEX.length) {
            renderComplete();
        } else {
            renderStep();
        }
    }

    private void showBlockedWrongTap() {
        TextView warning = new TextView(this);
        warning.setText("WRONG CARD TAPPED");
        warning.setTextColor(Color.RED);
        warning.setTextSize(22);
        warning.setGravity(Gravity.CENTER);
        warning.setContentDescription("visual-task-wrong-tap");
        root.addView(warning, 0, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(60)));
    }

    private void renderComplete() {
        LinearLayout complete = new LinearLayout(this);
        complete.setOrientation(LinearLayout.VERTICAL);
        complete.setGravity(Gravity.CENTER);
        complete.setPadding(dp(24), dp(24), dp(24), dp(24));
        complete.setBackgroundColor(Color.rgb(232, 248, 237));

        TextView result = new TextView(this);
        result.setText("VISUAL_TASK_COMPLETE");
        result.setContentDescription("visual-task-complete");
        result.setTextColor(Color.rgb(24, 112, 56));
        result.setTextSize(28);
        result.setTypeface(Typeface.DEFAULT_BOLD);
        result.setGravity(Gravity.CENTER);
        complete.addView(result, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(140)));

        TextView detail = new TextView(this);
        detail.setText("Three screenshot → plan → gated tap cycles completed.");
        detail.setTextColor(Color.DKGRAY);
        detail.setTextSize(18);
        detail.setGravity(Gravity.CENTER);
        complete.addView(detail, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(100)));

        Button finish = new Button(this);
        finish.setText("Return to ZhiBan Agent");
        finish.setOnClickListener(v -> finish());
        complete.addView(finish, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(70)));
        setContentView(complete);
    }

    private int colorIndex(String label) {
        if (label.startsWith("RED")) return 0;
        if (label.startsWith("GREEN")) return 1;
        if (label.startsWith("ORANGE")) return 2;
        if (label.startsWith("BLUE")) return 3;
        return 4;
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }
}
