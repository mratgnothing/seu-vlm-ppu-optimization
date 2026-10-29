const screens = [...document.querySelectorAll('.screen')];
const back = document.querySelector('#back');
let current = 'home';

function show(id) {
  current = id;
  screens.forEach(s => s.classList.toggle('active', s.id === id));
  back.classList.toggle('hidden', id === 'home');
  window.scrollTo(0, 0);
}

document.querySelectorAll('[data-open]').forEach(button => {
  button.addEventListener('click', () => show(button.dataset.open));
});
back.addEventListener('click', () => show('home'));

document.querySelectorAll('[data-contact]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelector('#recipient').textContent = button.dataset.contact === 'daughter'
      ? '陈小雨（女儿）' : '陈小雨（社区志愿者）';
    document.querySelector('#composer').classList.remove('hidden');
  });
});

document.querySelector('#previewMessage').addEventListener('click', () => {
  const recipient = document.querySelector('#recipient').textContent;
  const text = document.querySelector('#messageText').value.trim();
  if (!text) return;
  document.querySelector('#messagePreview').textContent = `将向 ${recipient} 发送：${text}`;
  document.querySelector('#messageConfirm').classList.remove('hidden');
});
document.querySelector('#confirmSend').addEventListener('click', () => {
  document.querySelector('#messageConfirm').classList.add('hidden');
  document.querySelector('#messageSuccess').classList.remove('hidden');
});
document.querySelector('#cancelSend').addEventListener('click', () => {
  document.querySelector('#messageConfirm').classList.add('hidden');
});

const tomorrow = new Date(Date.now() + 86400000);
document.querySelector('#eventDate').value = tomorrow.toISOString().slice(0, 10);
document.querySelector('#previewEvent').addEventListener('click', () => {
  const title = document.querySelector('#eventTitle').value;
  const date = document.querySelector('#eventDate').value;
  const time = document.querySelector('#eventTime').value;
  document.querySelector('#eventPreview').textContent = `${date} ${time}：${title}`;
  document.querySelector('#eventConfirm').classList.remove('hidden');
});
document.querySelector('#confirmEvent').addEventListener('click', () => {
  document.querySelector('#eventConfirm').classList.add('hidden');
  document.querySelector('#eventSuccess').classList.remove('hidden');
});

document.querySelectorAll('[data-font]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelector('#fontSample').style.fontSize = `${button.dataset.font}px`;
    document.querySelector('#fontSuccess').classList.remove('hidden');
  });
});
document.querySelector('#explainRisk').addEventListener('click', () => {
  document.querySelector('#riskResult').classList.remove('hidden');
});
document.querySelector('#payButton').addEventListener('click', () => {
  document.querySelector('#handoff').classList.remove('hidden');
  window.scrollTo(0, 0);
});
document.querySelector('#reset').addEventListener('click', () => location.reload());

const params = new URLSearchParams(location.search);
const initialScreen = params.get('screen');
if (initialScreen && document.querySelector(`#${initialScreen}`)) {
  show(initialScreen);
}
const initialContact = params.get('contact');
if (initialScreen === 'messages' && initialContact) {
  const contactButton = document.querySelector(`[data-contact="${initialContact}"]`);
  if (contactButton) contactButton.click();
}
