package ai.zhiman.accessibility;

import android.inputmethodservice.InputMethodService;
import android.view.inputmethod.InputConnection;

public final class AgentInputMethodService extends InputMethodService {
    private static volatile AgentInputMethodService instance;

    public static AgentInputMethodService getInstance() {
        return instance;
    }

    @Override
    public void onCreate() {
        super.onCreate();
        instance = this;
    }

    @Override
    public void onDestroy() {
        if (instance == this) {
            instance = null;
        }
        super.onDestroy();
    }

    @Override
    public boolean onEvaluateInputViewShown() {
        // This prototype acts as a headless text bridge after the user explicitly
        // selects it as the active IME. It does not draw its own keyboard UI.
        return false;
    }

    public synchronized boolean commitTextNow(CharSequence text) {
        InputConnection connection = getCurrentInputConnection();
        if (connection == null) {
            return false;
        }
        return connection.commitText(text == null ? "" : text, 1);
    }

    public synchronized boolean performEditorActionNow() {
        InputConnection connection = getCurrentInputConnection();
        if (connection == null || getCurrentInputEditorInfo() == null) {
            return false;
        }
        int action = getCurrentInputEditorInfo().imeOptions
                & android.view.inputmethod.EditorInfo.IME_MASK_ACTION;
        if (action == android.view.inputmethod.EditorInfo.IME_ACTION_NONE) {
            return false;
        }
        return connection.performEditorAction(action);
    }
}
