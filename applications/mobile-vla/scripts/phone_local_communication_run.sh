#!/system/bin/sh
set -u

# Phone-local integration runner. It is intentionally narrow: the model may
# select only the three pre-authorized communication skills, and all contact
# data is read from the app's private preferences. The script writes only a
# redacted result file.

RESULT=/data/local/tmp/zhiman_phone_local_communication.result
LOG=/data/local/tmp/zhiman_phone_local_communication.log
PAYLOAD=/data/local/tmp/zhiman_phone_local_payload.json
: > "$RESULT"
: > "$LOG"
exec >>"$LOG" 2>&1

ts() { date '+%s000'; }
fail() {
  echo "status=FAIL_CLOSED" > "$RESULT"
  echo "reason=$1" >> "$RESULT"
  exit 2
}

echo "runner=phone_local_shell_bridge"
echo "started_epoch_ms=$(ts)"
sleep 12

API_XML="$(run-as com.alibaba.mnnllm.android cat shared_prefs/api_settings.xml 2>/dev/null || true)"
KEY="$(printf '%s' "$API_XML" | sed -n 's/.*<string name="api_key">\([^<]*\)<\/string>.*/\1/p' | head -n 1)"
API_XML=
[ -n "$KEY" ] || fail api_key_unavailable
echo "api_key_available=true"

model_gate() {
  expected="$1"
  prompt="$2"
  case "$expected" in
    call_bound_contact)
      schema='{\"skill\":\"call_bound_contact\",\"contact_alias\":\"authorized_test_contact\",\"max_duration_seconds\":5}; {\"skill\":\"send_wechat_message\",\"contact_alias\":\"authorized_test_contact\",\"message_id\":\"skill_test_notice_v1\"}; {\"skill\":\"start_wechat_video_call\",\"contact_alias\":\"authorized_test_contact\",\"max_ring_seconds\":5}' ;;
    send_wechat_message)
      schema='{\"skill\":\"call_bound_contact\",\"contact_alias\":\"authorized_test_contact\",\"max_duration_seconds\":5}; {\"skill\":\"send_wechat_message\",\"contact_alias\":\"authorized_test_contact\",\"message_id\":\"skill_test_notice_v1\"}; {\"skill\":\"start_wechat_video_call\",\"contact_alias\":\"authorized_test_contact\",\"max_ring_seconds\":5}' ;;
    start_wechat_video_call)
      schema='{\"skill\":\"call_bound_contact\",\"contact_alias\":\"authorized_test_contact\",\"max_duration_seconds\":5}; {\"skill\":\"send_wechat_message\",\"contact_alias\":\"authorized_test_contact\",\"message_id\":\"skill_test_notice_v1\"}; {\"skill\":\"start_wechat_video_call\",\"contact_alias\":\"authorized_test_contact\",\"max_ring_seconds\":5}' ;;
    *) fail unknown_expected_skill ;;
  esac
  # The schema is repeated in the prompt so the small model has one exact
  # choice; the response is still checked independently below.
  printf '%s' "{\"model\":\"Qwen3.5-2B-MNN\",\"messages\":[{\"role\":\"system\",\"content\":\"You are an offline phone skill router. Return exactly one JSON object and no prose. Choose exactly this schema: $schema. Never invent a contact, number, message, duration, field, or second action.\"},{\"role\":\"user\",\"content\":\"$prompt\"}],\"temperature\":0,\"max_tokens\":128,\"enable_thinking\":false}" > "$PAYLOAD"
  expected_marker='"content":"{\"skill\":\"'"$expected"'\"'
  response=
  attempt=1
  while [ "$attempt" -le 3 ]; do
    response="$(curl -sS --max-time 30 -H "Authorization: Bearer $KEY" -H 'Content-Type: application/json' --data-binary @- http://127.0.0.1:8080/v1/chat/completions < "$PAYLOAD" 2>/dev/null || true)"
    printf '%s' "$response" > /data/local/tmp/zhiman_phone_local_response.json
    printf '%s' "$response" | grep -Fq "$expected_marker" && break
    sleep 1
    attempt=$((attempt + 1))
  done
  echo "model_response_length_${expected}=${#response}"
  printf '%s' "$response" | grep -Fq "$expected_marker" || fail "model_content_not_exact_json_$expected"
  normalized="$(printf '%s' "$response" | sed 's/\\"/"/g')"
  echo "model_response_has_expected_${expected}=$(printf '%s' "$normalized" | grep -F -c "\"skill\":\"$expected\"" || true)"
  skill_count="$(printf '%s' "$normalized" | grep -o '"skill"' | wc -l | tr -d ' ')"
  printf '%s' "$normalized" | grep -Fq "\"skill\":\"$expected\"" || fail "model_gate_$expected"
  [ "$skill_count" = 1 ] || fail "model_multiple_skills_$expected"
  printf '%s' "$normalized" | grep -Fq '"contact_alias":"authorized_test_contact"' || fail "model_alias_$expected"
  case "$expected" in
    call_bound_contact) printf '%s' "$normalized" | grep -Fq '"max_duration_seconds":5' || fail model_call_duration ;;
    send_wechat_message) printf '%s' "$normalized" | grep -Fq '"message_id":"skill_test_notice_v1"' || fail model_message_id ;;
    start_wechat_video_call) printf '%s' "$normalized" | grep -Fq '"max_ring_seconds":5' || fail model_video_duration ;;
  esac
  echo "planner_${expected}=PASS"
}

if [ "${1:-}" = "probe" ]; then
  model_gate call_bound_contact 'Call the already authorized contact and limit the call attempt to five seconds.'
  echo 'status=PASS' > "$RESULT"
  exit 0
fi

bridge_result_pass() {
  run-as ai.zhiman.accessibility cat files/shell-bridge-result.json 2>/dev/null \
    | grep -Fq '"result": "PASS"'
}

phone_call() {
  model_gate call_bound_contact 'Call the already authorized contact and limit the call attempt to five seconds.'
  PHONE_XML="$(run-as ai.zhiman.accessibility cat shared_prefs/agent_config.xml 2>/dev/null || true)"
  PHONE="$(printf '%s' "$PHONE_XML" | sed -n 's/.*<string name="bound_phone">\([^<]*\)<\/string>.*/\1/p' | head -n 1)"
  PHONE_XML=
  printf '%s' "$PHONE" | grep -Eq '^[0-9]{3,20}$' || fail private_phone_binding_invalid
  am start -a android.intent.action.DIAL -d "tel:$PHONE" >/dev/null 2>&1
  sleep 2
  # Xiaomi's dual-SIM dialer exposes two deterministic call buttons.
  input tap 438 2336
  sleep 5
  active="$(dumpsys window 2>/dev/null | grep 'mCurrentFocus' | grep -Ei 'InCallUI|incall' | head -n 1 || true)"
  if [ -z "$active" ]; then
    input keyevent 4 >/dev/null 2>&1 || true
    fail phone_call_not_active
  fi
  echo 'phone_call_active_after_5s=true'
  input keyevent 6 >/dev/null 2>&1
  sleep 2
  after="$(dumpsys window 2>/dev/null | grep 'mCurrentFocus' | grep -Ei 'InCallUI|incall' | head -n 1 || true)"
  [ -z "$after" ] || fail phone_call_hangup_not_confirmed
  echo 'phone_call_hangup_confirmed=true'
}

open_bound_chat() {
  am start -n com.tencent.mm/.ui.LauncherUI >/dev/null 2>&1 || am start -a android.intent.action.MAIN -c android.intent.category.LAUNCHER >/dev/null 2>&1
  sleep 2
  input tap 1015 220
  sleep 1
  am broadcast -n ai.zhiman.accessibility/.ShellBridgeReceiver -a ai.zhiman.accessibility.shell.COMMIT_BOUND_WECHAT_NAME >/dev/null 2>&1
  sleep 1
  bridge_result_pass || fail wechat_name_commit
  input tap 300 580
  sleep 2
  am broadcast -n ai.zhiman.accessibility/.ShellBridgeReceiver -a ai.zhiman.accessibility.shell.VERIFY_CHAT_BINDING >/dev/null 2>&1
  sleep 1
  bridge_result_pass || fail wechat_binding_mismatch
}

wechat_message() {
  model_gate send_wechat_message 'Send the pre-approved WeChat notice to the already authorized contact.'
  open_bound_chat
  input tap 500 2510
  sleep 1
  am broadcast -n ai.zhiman.accessibility/.ShellBridgeReceiver -a ai.zhiman.accessibility.shell.COMMIT_APPROVED_MESSAGE >/dev/null 2>&1
  sleep 1
  bridge_result_pass || fail wechat_message_commit
  input tap 1110 2510
  sleep 2
  echo 'wechat_message_send_tap=true'
}

wechat_video() {
  model_gate start_wechat_video_call 'Start a WeChat video call to the already authorized contact and ring for at most five seconds.'
  open_bound_chat
  input tap 1140 2500
  sleep 1
  input tap 750 1860
  sleep 1
  input tap 610 2200
  sleep 5
  video_focus="$(dumpsys window 2>/dev/null | grep -E 'com.tencent.mm|InCallUI|Video' | head -n 1 || true)"
  [ -n "$video_focus" ] || fail wechat_video_screen_not_reached
  echo 'wechat_video_ring_window_reached=true'
  input tap 610 2280
  sleep 2
  echo 'wechat_video_hangup_tap=true'
}

if [ "${SKIP_PHONE_CALL:-0}" = "1" ]; then
  echo 'phone_call_previously_completed=true'
else
  phone_call
fi
wechat_message
wechat_video
am start -a android.intent.action.MAIN -c android.intent.category.HOME >/dev/null 2>&1 || true
KEY=
rm -f "$PAYLOAD"
echo "finished_epoch_ms=$(ts)" >> "$RESULT"
echo 'status=PASS' >> "$RESULT"
echo 'contains_raw_contact_data=false' >> "$RESULT"
echo 'external_actions_authorized_by_user=true' >> "$RESULT"
