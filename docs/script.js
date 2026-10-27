'use strict';

document.querySelectorAll('.compare').forEach(panel => {
  const slider = panel.querySelector('input');
  const update = () => {
    panel.style.setProperty('--split', `${slider.value}%`);
    slider.setAttribute('aria-valuetext', `${slider.value}% LR visible`);
  };
  slider.addEventListener('input', update);
  // Map mouse, pen, and touch positions to the same aligned image coordinates.
  slider.addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    slider.focus({preventScroll: true});
    slider.setPointerCapture(event.pointerId);
    move(event);
  });
  const move = event => {
    if (!slider.hasPointerCapture(event.pointerId)) return;
    const rect = panel.getBoundingClientRect();
    slider.value = Math.round(Math.max(0, Math.min(100, (event.clientX - rect.left) / rect.width * 100)));
    update();
  };
  slider.addEventListener('pointermove', move);
  slider.addEventListener('pointerup', event => {
    if (slider.hasPointerCapture(event.pointerId)) slider.releasePointerCapture(event.pointerId);
  });
  update();
});

const lightbox = document.querySelector('#lightbox');
document.querySelectorAll('a.zoom').forEach(link => {
  link.addEventListener('click', event => {
    if (!lightbox.showModal || event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return;
    event.preventDefault();
    lightbox.querySelector('img').src = link.href;
    lightbox.querySelector('img').alt = link.querySelector('img').alt;
    lightbox.querySelector('p').textContent = link.dataset.caption || '';
    lightbox.showModal();
  });
});
lightbox.querySelector('button').addEventListener('click', () => lightbox.close());
lightbox.addEventListener('click', event => { if (event.target === lightbox) lightbox.close(); });

document.querySelector('#copy-citation').addEventListener('click', async () => {
  const status = document.querySelector('#copy-status');
  try {
    await navigator.clipboard.writeText(document.querySelector('#bibtex').textContent);
    status.textContent = 'BibTeX copied.';
  } catch {
    const selection = window.getSelection();
    const range = document.createRange();
    range.selectNodeContents(document.querySelector('#bibtex'));
    selection.removeAllRanges();
    selection.addRange(range);
    status.textContent = 'Citation selected. Press Ctrl+C (or ⌘C) to copy.';
  }
});
