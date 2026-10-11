const copyButton = document.querySelector('#copy-citation');
const citation = document.querySelector('#citation');
const copyStatus = document.querySelector('#copy-status');

copyButton.hidden = false;
copyButton.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(citation.textContent);
    copyStatus.textContent = 'Citation copied.';
  } catch {
    const range = document.createRange();
    range.selectNodeContents(citation);
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    copyStatus.textContent = 'Citation selected. Press Ctrl+C or ⌘C to copy.';
  }
});
