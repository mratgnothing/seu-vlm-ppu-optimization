#!/system/bin/sh
set -u

RESULT=/data/local/tmp/zhiman_phone_local_wechat_actions.result
: > "$RESULT"
bridge_pass() {
  run-as ai.zhiman.accessibility cat files/shell-bridge-result.json 2>/dev/null \
    | grep -Fq '"result": "PASS"'
}
open_bound_chat() {
  am start -n com.tencent.mm/.ui.LauncherUI >/dev/null 2>&1
  sleep 2
  input tap 1015 220
  sleep 1
  am broadcast -n ai.zhiman.accessibility/.ShellBridgeReceiver -a ai.zhiman.accessibility.shell.COMMIT_BOUND_WECHAT_NAME >/dev/null 2>&1
  sleep 1
  bridge_pass || exit 11
  input tap 300 580
  sleep 2
  am broadcast -n ai.zhiman.accessibility/.ShellBridgeReceiver -a ai.zhiman.accessibility.shell.VERIFY_CHAT_BINDING >/dev/null 2>&1
  sleep 1
  bridge_pass || exit 12
}

if [ "${1:-}" = "message" ]; then
  [ -f /data/local/tmp/send_plan.pass ] || exit 13
  open_bound_chat
  input tap 500 2510
  sleep 1
  am broadcast -n ai.zhiman.accessibility/.ShellBridgeReceiver -a ai.zhiman.accessibility.shell.COMMIT_APPROVED_MESSAGE >/dev/null 2>&1
  sleep 1
  bridge_pass || exit 14
  input tap 1110 2510
  sleep 2
  echo 'wechat_binding_verified=true' >> "$RESULT"
  echo 'message_commit_verified=true' >> "$RESULT"
  echo 'message_send_tap=true' >> "$RESULT"
  echo 'status=PASS' >> "$RESULT"
  exit 0
fi

if [ "__ARG1__" = "video" ]; then
  [ -f /data/local/tmp/video_plan.pass ] || exit 15
  open_bound_chat
  input tap 1140 2500
  sleep 1
  input tap 750 1860
  sleep 1
  input tap 610 2200
  sleep 5
  input tap 610 2280
  sleep 2
  echo 'wechat_binding_verified=true' >> "$RESULT"
  echo 'video_ring_window_5s=true' >> "$RESULT"
  echo 'video_hangup_tap=true' >> "$RESULT"
  echo 'status=PASS' >> "$RESULT"
  exit 0
fi
exit 10
