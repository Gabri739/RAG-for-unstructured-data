// Interfaccia: una chat per ogni PDF, domande e fonti.
// Tutto il testo che arriva da PDF e LLM viene inserito con textContent, mai come HTML.

const $ = (id) => document.getElementById(id);
const docList = $('docList');
const layout = $('layout');
const collapseBtn = $('collapseBtn');
const uploadBtn = $('uploadBtn');
const uploadLabel = $('uploadLabel');
const fileInput = $('fileInput');
const codeOptions = $('codeOptions');
const docTitle = $('docTitle');
const pdfLink = $('pdfLink');
const banner = $('banner');
const messages = $('messages');
const thread = $('thread');
const form = $('form');
const question = $('question');
const sendBtn = $('sendBtn');
const lockBtn = $('lockBtn');
const lockTip = $('lockTip');
const confirmDialog = $('confirmDialog');
const confirmCancel = $('confirmCancel');
const confirmOk = $('confirmOk');
const confirmError = $('confirmError');

const state = {
    status: null,
    docs: [],
    current: null, // id del documento selezionato
    threads: new Map(), // id -> elementi della conversazione
    asking: false,
    deleting: null, // documento nella finestra di conferma
    lastStatus: new Map(), // id -> stato visto all'ultimo aggiornamento
    justReady: new Map(), // id -> fino a quando mostrare le barre complete
    uploading: false,
    notice: '',
};

// ---------------------------------------------------------------- utilità
function el(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

async function api(path, options) {
    let response;
    try {
        response = await fetch(path, options);
    } catch {
        throw new Error('Il server non risponde. Controlla che sia avviato.');
    }
    if (response.status === 204) return null;
    const data = await response.json().catch(() => null);
    if (!response.ok) {
        const detail = data && data.detail;
        throw new Error(typeof detail === 'string' ? detail : `Richiesta non riuscita (${response.status}).`);
    }
    return data;
}

const currentDoc = () => state.docs.find((d) => d.id === state.current);

function pdfUrl(doc, page) {
    return `/api/documents/${encodeURIComponent(doc.id)}/pdf${page ? `#page=${page}` : ''}`;
}

// ---------------------------------------------------------------- stato
function bannerText() {
    if (state.notice) return state.notice;
    const s = state.status;
    if (!s) return '';
    if (s.state === 'error') return s.error;
    if (s.state === 'loading') return 'Carico i modelli: può richiedere un minuto.';
    const doc = currentDoc();
    if (s.ingesting && doc && doc.status === 'processing') return ''; // lo dicono già le barre
    if (s.ingesting) return 'Sto elaborando un documento: le domande riprendono appena finisce.';
    if (s.llm === 'unreachable') return 'Ollama non è raggiungibile: avvia l\'app Ollama.';
    if (s.llm === 'missing_model') return `Manca il modello: esegui "ollama pull ${s.model}".`;
    return '';
}

function render() {
    const text = bannerText();
    banner.textContent = text;
    banner.hidden = !text;

    const s = state.status;
    const ready = Boolean(s && s.state === 'ready');
    const doc = currentDoc();
    const canAsk = ready && !s.ingesting && s.llm === 'ready' && doc && doc.status === 'ready';
    uploadBtn.disabled = !ready || s.ingesting || state.uploading;
    uploadLabel.textContent = state.uploading ? 'Caricamento…' : 'Nuova chat con un PDF';
    uploadBtn.title = uploadLabel.textContent;
    question.disabled = !doc || doc.status !== 'ready';
    sendBtn.disabled = !canAsk || state.asking || !question.value.trim();

    // Con un modello cloud di Ollama gli estratti uscirebbero dal PC: il lucchetto sarebbe falso
    lockBtn.hidden = !s || !s.model || /cloud$/.test(s.model);

    docTitle.textContent = doc ? doc.title : 'Nessuna chat aperta';
    pdfLink.hidden = !doc || !doc.has_pdf;
    if (doc && doc.has_pdf) pdfLink.href = pdfUrl(doc);
}

function docInfo(doc) {
    if (doc.status === 'error') return doc.message || 'Elaborazione fallita.';
    if (doc.status === 'processing') {
        const phase = doc.progress && doc.progress.phases.find((p) => p.state === 'active');
        if (!phase) return 'In coda…';
        return phase.percent === null ? `${phase.name}…` : `${phase.name} ${phase.percent}%`;
    }
    const parts = [];
    if (doc.pages) parts.push(`${doc.pages} ${doc.pages === 1 ? 'pagina' : 'pagine'}`);
    parts.push(`${doc.chunks} estratti`);
    if (doc.message) parts.push(doc.message);
    return parts.join(' · ');
}

const initials = (title) => [...title.replace(/[^\p{L}\p{N}]/gu, '')].slice(0, 2).join('') || '?';

const TRASH = ['M3 6h18', 'M8 6V4a1 1 0 0 1 1-1h6a1 1 0 0 1 1 1v2', 'M19 6l-1 14a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1L5 6', 'M10 11v6', 'M14 11v6'];

function trashIcon() {
    const ns = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(ns, 'svg');
    const attributes = { width: 18, height: 18, viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', 'stroke-width': 2, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', 'aria-hidden': 'true' };
    for (const [name, value] of Object.entries(attributes)) svg.setAttribute(name, value);
    for (const d of TRASH) {
        const path = document.createElementNS(ns, 'path');
        path.setAttribute('d', d);
        svg.append(path);
    }
    return svg;
}

function deleteButton(doc) {
    const button = el('button', 'delete');
    button.type = 'button';
    button.title = 'Elimina chat';
    button.setAttribute('aria-label', `Elimina la chat ${doc.title}`);
    button.append(trashIcon());
    button.addEventListener('click', () => askDelete(doc));
    return button;
}

let renderedDocs = '';
const infoNodes = new Map();
function renderDocs() {
    // L'avanzamento cambia di continuo: aggiorna solo il testo, senza ricostruire l'elenco
    const key = JSON.stringify([state.docs.map((d) => ({ ...d, progress: null })), state.current]);
    if (key === renderedDocs) {
        for (const doc of state.docs) {
            const info = infoNodes.get(doc.id);
            if (info && info.textContent !== docInfo(doc)) info.textContent = docInfo(doc);
        }
        return;
    }
    renderedDocs = key;
    infoNodes.clear();
    docList.replaceChildren();
    if (!state.docs.length) {
        docList.append(el('li', 'empty-note', 'Nessuna chat: aggiungi il primo PDF.'));
        return;
    }
    for (const doc of state.docs) {
        const item = el('li');
        const button = el('button', doc.status === 'error' ? 'doc failed' : 'doc');
        button.type = 'button';
        button.title = doc.title;
        if (doc.id === state.current) button.setAttribute('aria-current', 'true');
        const info = el('span', doc.status === 'error' ? 'doc-info error' : 'doc-info', docInfo(doc));
        infoNodes.set(doc.id, info);
        button.append(el('span', 'doc-initials', initials(doc.title)), el('span', 'doc-title', doc.title), info);
        button.addEventListener('click', () => select(doc.id));
        item.append(button);
        if (doc.status !== 'processing') item.append(deleteButton(doc));
        docList.append(item);
    }
}

function renderCodes() {
    if (codeOptions.childElementCount || !state.status) return;
    for (const name of state.status.reference_codes) {
        const label = el('label', 'option');
        const input = el('input');
        input.type = 'checkbox';
        input.value = name;
        label.append(input, el('span', '', `Riconosci i codici ${name.toUpperCase()} (per i documenti che li usano)`));
        codeOptions.append(label);
    }
}

let noticeTimer;
function notify(text) {
    state.notice = text;
    clearTimeout(noticeTimer);
    noticeTimer = setTimeout(() => {
        state.notice = '';
        render();
    }, 8000);
    render();
}

// Un solo ciclo di aggiornamento: ogni chiamata annulla l'attesa precedente
let pollTimer;
let pollRun = 0;
async function refresh() {
    const run = ++pollRun;
    clearTimeout(pollTimer);
    try {
        const [status, docs] = await Promise.all([api('/api/status'), api('/api/documents')]);
        if (run !== pollRun) return;
        state.status = status;
        state.docs = docs;
        for (const doc of docs) {
            if (state.lastStatus.get(doc.id) === 'processing' && doc.status === 'ready') {
                state.justReady.set(doc.id, Date.now() + READY_MS);
                setTimeout(() => {
                    if (state.current === doc.id && thread.querySelector('.intro')) showIntro();
                }, READY_MS + 50);
            }
            state.lastStatus.set(doc.id, doc.status);
        }
        if (!currentDoc()) {
            const first = docs.find((d) => d.status === 'ready') || docs[0];
            if ((first ? first.id : null) !== state.current) select(first ? first.id : null);
        }
        renderCodes();
        renderDocs();
        render();
        if (!thread.childElementCount || thread.querySelector('.intro')) showIntro();
    } catch (error) {
        if (run !== pollRun) return;
        state.status = { state: 'error', error: error.message, reference_codes: [] };
        render();
    }
    const s = state.status;
    const active = s.state !== 'ready' || s.ingesting || s.llm !== 'ready';
    pollTimer = setTimeout(refresh, s.ingesting ? 700 : active ? 2000 : 15000);
}

// ---------------------------------------------------------------- documenti
const READY_MS = 2500;
const STATE_TEXT = { waiting: 'In attesa', active: 'In corso', done: 'Completata', error: 'Non riuscita' };

// Pannello con una barra per fase, l'azione in corso e i passaggi conclusi
function ingestPanel(doc) {
    const panel = el('section', 'intro ingest');
    panel.setAttribute('aria-live', 'off');
    const title = el('h2');
    const action = el('p', 'ingest-action');
    action.setAttribute('aria-live', 'polite');
    const time = el('span', 'ingest-time');
    const head = el('div', 'ingest-head');
    head.append(action, time);
    const phases = el('ul', 'ingest-phases');
    const rows = new Map();
    for (const phase of doc.progress.phases) {
        const row = el('li', 'phase');
        const mark = el('span', 'phase-mark');
        const name = el('span', 'phase-name', phase.name);
        const detail = el('span', 'phase-detail');
        const value = el('span', 'phase-value');
        const bar = el('div', 'phase-bar');
        bar.setAttribute('role', 'progressbar');
        bar.setAttribute('aria-label', phase.name);
        bar.setAttribute('aria-valuemin', '0');
        bar.setAttribute('aria-valuemax', '100');
        const fill = el('div', 'phase-fill');
        bar.append(fill);
        row.append(mark, name, detail, value, bar);
        phases.append(row);
        rows.set(phase.id, { row, mark, detail, value, bar, fill });
    }
    const logTitle = el('p', 'ingest-log-title', 'Passaggi completati');
    const log = el('ol', 'ingest-log');
    panel.append(title, head, phases, logTitle, log);
    panel.ingest = { id: doc.id, title, action, time, rows, log, logTitle };
    return panel;
}

const MARKS = { waiting: '', active: '', done: '✓', error: '!' };

function updateIngest(panel, doc) {
    const { title, action, time, rows, log, logTitle } = panel.ingest;
    const progress = doc.progress;
    const failed = doc.status === 'error';
    title.textContent = failed ? 'Elaborazione non riuscita' : doc.status === 'ready' ? 'Chat pronta' : 'Preparo la chat';
    const text = doc.status === 'processing' ? `${progress.action}…` : progress.action;
    if (action.textContent !== text) action.textContent = text;
    action.classList.toggle('error', failed);
    time.textContent = `${Math.round(progress.seconds)} s`;
    for (const phase of progress.phases) {
        const { row, mark, detail, value, bar, fill } = rows.get(phase.id);
        const unknown = phase.percent === null;
        row.className = `phase is-${phase.state}${unknown && phase.state === 'active' ? ' is-unknown' : ''}`;
        mark.textContent = MARKS[phase.state];
        detail.textContent = phase.detail;
        value.textContent = phase.state === 'active' && !unknown ? `${phase.percent}%` : phase.state === 'done' ? '100%' : STATE_TEXT[phase.state];
        bar.setAttribute('aria-valuetext', `${STATE_TEXT[phase.state]}${unknown ? '' : `, ${phase.percent}%`}`);
        if (unknown) bar.removeAttribute('aria-valuenow');
        else bar.setAttribute('aria-valuenow', String(phase.percent));
        fill.style.transform = `scaleX(${unknown ? (phase.state === 'active' ? 1 : 0) : phase.percent / 100})`;
    }
    for (const entry of progress.log.slice(log.childElementCount)) {
        const item = el('li');
        item.append(el('span', '', entry.text), el('span', 'log-time', `${entry.seconds.toLocaleString('it-IT')} s`));
        log.prepend(item); // il più recente in cima
    }
    logTitle.hidden = !log.childElementCount;
}

function showIntro() {
    const doc = currentDoc();
    const ready = doc && doc.status === 'ready';
    if (doc && doc.progress && (!ready || (state.justReady.get(doc.id) || 0) > Date.now())) {
        let panel = thread.querySelector('.ingest');
        if (!panel || panel.ingest.id !== doc.id) {
            panel = ingestPanel(doc);
            thread.replaceChildren(panel);
        }
        updateIngest(panel, doc);
        return;
    }
    let title = 'Aggiungi un PDF';
    let text = 'Quando è pronto potrai fare domande sul suo contenuto.';
    if (doc && doc.status === 'ready') {
        title = 'Cosa vuoi sapere?';
        text = 'La risposta usa solo il contenuto del documento e cita le pagine. Se l\'informazione manca, te lo dico.';
    } else if (doc && doc.status === 'processing') {
        title = 'Preparo la chat';
        text = 'In elaborazione…';
    } else if (doc) {
        title = 'Documento non disponibile';
        text = doc.message || 'Elaborazione fallita.';
    }
    const current = thread.querySelector('.intro');
    if (current && current.dataset.key === title + text) return;
    const intro = el('div', 'intro');
    intro.dataset.key = title + text;
    intro.append(el('h2', '', title), el('p', '', text));
    thread.replaceChildren(intro);
}

function select(id) {
    if (id === state.current && thread.childElementCount) return;
    if (state.current) state.threads.set(state.current, [...thread.childNodes]);
    state.current = id;
    thread.replaceChildren(...(state.threads.get(id) || []));
    if (!thread.childElementCount) showIntro();
    renderDocs();
    render();
    if (id && !question.disabled) question.focus();
}

async function upload(file) {
    if (uploadBtn.disabled) return;
    if (!/\.pdf$/i.test(file.name)) {
        notify('Sono accettati solo file PDF.');
        return;
    }
    state.uploading = true;
    render();
    const data = new FormData();
    data.append('file', file);
    for (const input of codeOptions.querySelectorAll('input:checked')) data.append('reference_codes', input.value);
    let created = null;
    try {
        created = await api('/api/documents', { method: 'POST', body: data });
    } catch (error) {
        notify(error.message);
    }
    state.uploading = false;
    await refresh();
    if (created) select(created.id); // la nuova chat si apre subito, e si attiva quando il PDF è pronto
}

// ---------------------------------------------------------------- lucchetto
function showLockTip() {
    lockTip.hidden = false;
    const anchor = lockBtn.getBoundingClientRect();
    const tip = lockTip.getBoundingClientRect();
    const left = Math.min(anchor.left, window.innerWidth - tip.width - 16);
    const above = anchor.top - tip.height - 8;
    lockTip.style.left = `${Math.max(16, left)}px`;
    lockTip.style.top = `${above >= 8 ? above : anchor.bottom + 8}px`;
}

function hideLockTip() {
    lockTip.hidden = true;
}

lockBtn.addEventListener('mouseenter', showLockTip);
lockBtn.addEventListener('mouseleave', hideLockTip);
lockBtn.addEventListener('focus', showLockTip);
lockBtn.addEventListener('blur', hideLockTip);
lockBtn.addEventListener('click', showLockTip); // telefono: un tocco
window.addEventListener('resize', hideLockTip);
document.addEventListener('keydown', (event) => {
    if (event.key === 'Escape') hideLockTip();
});

// ---------------------------------------------------------------- eliminazione
function askDelete(doc) {
    state.deleting = doc;
    $('confirmDoc').textContent = doc.title;
    $('confirmNote').textContent = doc.managed
        ? 'Il PDF caricato e le risposte di questa chat vengono cancellati. L\'operazione non si può annullare.'
        : 'Questo documento non è stato caricato da qui: viene tolto dall\'app, ma i suoi file restano sul disco.';
    confirmError.hidden = true;
    confirmDialog.showModal();
}

function deletingBusy(busy) {
    confirmOk.disabled = busy;
    confirmCancel.disabled = busy;
    confirmOk.textContent = busy ? 'Elimino…' : 'Elimina';
}

confirmOk.addEventListener('click', async () => {
    const doc = state.deleting;
    if (!doc) return;
    deletingBusy(true);
    confirmError.hidden = true;
    let failure = '';
    try {
        await api(`/api/documents/${encodeURIComponent(doc.id)}`, { method: 'DELETE' });
    } catch (error) {
        failure = error.message;
    }
    await refresh();
    deletingBusy(false);
    if (failure) {
        confirmError.textContent = failure;
        confirmError.hidden = false;
        return;
    }
    state.threads.delete(doc.id);
    state.lastStatus.delete(doc.id);
    state.justReady.delete(doc.id);
    confirmDialog.close();
});
confirmCancel.addEventListener('click', () => confirmDialog.close());
confirmDialog.addEventListener('cancel', (event) => {
    if (confirmOk.disabled) event.preventDefault(); // non si chiude a metà operazione
});
confirmDialog.addEventListener('click', (event) => {
    if (event.target !== confirmDialog || confirmOk.disabled) return;
    const box = confirmDialog.getBoundingClientRect();
    const inside = event.clientX >= box.left && event.clientX <= box.right
        && event.clientY >= box.top && event.clientY <= box.bottom;
    if (!inside) confirmDialog.close(); // clic sullo sfondo
});
confirmDialog.addEventListener('close', () => {
    state.deleting = null;
});

// ---------------------------------------------------------------- barra laterale
function setCollapsed(collapsed) {
    layout.classList.toggle('collapsed', collapsed);
    const label = collapsed ? 'Espandi la barra laterale' : 'Riduci la barra laterale';
    collapseBtn.setAttribute('aria-expanded', String(!collapsed));
    collapseBtn.setAttribute('aria-label', label);
    collapseBtn.title = label;
}

function savedCollapsed() {
    try {
        return localStorage.getItem('sidebar-collapsed') === '1';
    } catch {
        return false;
    }
}

collapseBtn.addEventListener('click', () => {
    const collapsed = !layout.classList.contains('collapsed');
    setCollapsed(collapsed);
    try {
        localStorage.setItem('sidebar-collapsed', collapsed ? '1' : '0');
    } catch {
        // Senza memoria del browser la barra torna aperta al prossimo avvio
    }
});
setCollapsed(savedCollapsed());

uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) upload(fileInput.files[0]);
    fileInput.value = '';
});
document.addEventListener('dragover', (event) => {
    event.preventDefault();
    document.body.classList.add('dropping');
});
document.addEventListener('dragleave', (event) => {
    if (!event.relatedTarget) document.body.classList.remove('dropping');
});
document.addEventListener('drop', (event) => {
    event.preventDefault();
    document.body.classList.remove('dropping');
    if (event.dataTransfer.files[0]) upload(event.dataTransfer.files[0]);
});

// ---------------------------------------------------------------- domande
const CITATION = /\[\s*pagg?(?:\.|in[ae])?\s*(\d+)[^\]]*\]/gi;

function withCitations(text, doc) {
    const paragraph = el('p', 'reply-text');
    let last = 0;
    for (const match of text.matchAll(CITATION)) {
        paragraph.append(text.slice(last, match.index));
        const cite = el(doc.has_pdf ? 'a' : 'span', 'cite', match[0].slice(1, -1).trim());
        if (doc.has_pdf) {
            cite.href = pdfUrl(doc, match[1]);
            cite.target = '_blank';
            cite.rel = 'noopener';
        }
        paragraph.append(cite);
        last = match.index + match[0].length;
    }
    paragraph.append(text.slice(last));
    return paragraph;
}

function sourceNode(source, doc) {
    const node = el('details', source.cited ? 'source cited' : 'source');
    const title = `Pag. ${source.page_range}${source.section ? ` · ${source.section}` : ''}`;
    const summary = el('summary', '', title);
    if (source.type === 'table') summary.append(el('span', 'tag', 'tabella'));
    if (source.cited) summary.append(el('span', 'tag', 'pagina citata'));
    node.append(summary);
    // Contenuto e immagini si creano solo all'apertura
    node.addEventListener('toggle', () => {
        if (node.childElementCount > 1) return;
        for (const url of source.images) {
            const image = el('img');
            image.src = url;
            image.alt = `Tabella nelle pagine ${source.page_range}`;
            image.addEventListener('error', () => image.remove());
            node.append(image);
        }
        node.append(el('pre', '', source.content));
        if (doc.has_pdf && source.pages.length) {
            const link = el('a', 'cite', `Apri il PDF a pagina ${source.pages[0]}`);
            link.href = pdfUrl(doc, source.pages[0]);
            link.target = '_blank';
            link.rel = 'noopener';
            node.append(link);
        }
    });
    return node;
}

function replyNode(data, doc) {
    const node = el('div', data.refused ? 'reply refused' : 'reply');
    node.append(withCitations(data.answer, doc));
    if (data.truncated) node.append(el('p', 'reply-note warn', 'La risposta o gli estratti potrebbero essere incompleti.'));
    if (data.sources.length) {
        const sources = el('details', 'sources');
        sources.append(el('summary', '', `${data.refused ? 'Estratti consultati' : 'Fonti'} (${data.sources.length})`));
        for (const source of data.sources) sources.append(sourceNode(source, doc));
        node.append(sources);
    }
    return node;
}

async function ask() {
    const text = question.value.trim();
    const doc = currentDoc();
    if (sendBtn.disabled || !text || !doc) return;
    state.asking = true;
    question.value = '';
    resize();

    const waiting = el('div', 'reply waiting');
    waiting.append(el('i'), el('i'), el('i'), el('span', '', 'Cerco nel documento…'));
    thread.querySelector('.intro')?.remove();
    thread.append(el('div', 'question', text), waiting);
    messages.scrollTop = messages.scrollHeight;
    render();

    let node;
    try {
        const data = await api('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ doc_id: doc.id, question: text }),
        });
        node = replyNode(data, doc);
    } catch (error) {
        node = el('div', 'reply failed');
        node.append(el('p', 'reply-text', error.message));
        if (!question.value) question.value = text; // la domanda non va persa
    }
    // Se nel frattempo è stato scelto un altro documento, la risposta resta nella sua conversazione
    if (waiting.isConnected) {
        waiting.replaceWith(node);
        node.scrollIntoView({ block: 'nearest' });
    } else {
        const saved = state.threads.get(doc.id) || [];
        state.threads.set(doc.id, saved.map((n) => (n === waiting ? node : n)));
    }
    state.asking = false;
    render();
    refresh();
}

function resize() {
    question.style.height = 'auto';
    question.style.height = `${Math.min(question.scrollHeight + 2, 180)}px`;
}

question.addEventListener('input', () => {
    resize();
    render();
});
question.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        ask();
    }
});
form.addEventListener('submit', (event) => {
    event.preventDefault();
    ask();
});

refresh();
