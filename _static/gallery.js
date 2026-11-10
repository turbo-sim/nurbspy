/* Only the active card owns a WebGL viewer. All other cards stay as previews. */
(() => {
    function initialize() {
        let active = null;
        function close() {
            if (!active) return;
            active.viewer.replaceChildren(); // Unload the frame and its renderer.
            active.viewer.hidden = true;
            active.card.classList.remove('is-active');
            active.preview.hidden = false;
            active.preview.setAttribute('aria-expanded', 'false');
            active = null;
        }
        document.querySelectorAll('.nrb-demo-preview').forEach(preview => {
            preview.setAttribute('aria-expanded', 'false');
            preview.addEventListener('click', () => {
                close();
                const card = preview.closest('.nrb-demo-card');
                const viewer = card.querySelector('.nrb-demo-viewer');
                const toolbar = document.createElement('div');
                toolbar.className = 'nrb-demo-viewer-toolbar';
                const back = document.createElement('button');
                back.type = 'button';
                back.textContent = 'Close interactive figure';
                back.addEventListener('click', () => { close(); preview.focus(); });
                toolbar.append(back);
                const frame = document.createElement('iframe');
                frame.title = 'Interactive ' + preview.dataset.demoTitle;
                frame.src = preview.dataset.demoHtml;
                frame.allowFullscreen = true;
                viewer.append(toolbar, frame);
                viewer.hidden = false;
                preview.hidden = true;
                preview.setAttribute('aria-expanded', 'true');
                card.classList.add('is-active');
                active = { card, viewer, preview };
                viewer.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
                back.focus({ preventScroll: true });
            });
        });
    }
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', initialize);
    else initialize();
})();
