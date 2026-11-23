// Lente · cliente. Módulo único, sin dependencias ni paso de compilación.
// Rutas (hash): #/  ·  #/buscar?q=&m=&n=  ·  #/comparar?q=&n=  ·  #/evaluacion

const API = document.querySelector('meta[name="lente-api"]')?.content.replace(/\/$/, '') ?? '';
const METHODS = [
  { id: 'jaccard', label: 'Jaccard', family: 'Binario', short: 'Jac' },
  { id: 'tfidf', label: 'TF-IDF', family: 'Vectorial', short: 'TF' },
  { id: 'bm25', label: 'BM25', family: 'Probabilístico', short: 'BM' },
  { id: 'semantic', label: 'Semántico', family: 'Embeddings', short: 'Sem' },
];
const BY_ID = Object.fromEntries(METHODS.map((m) => [m.id, m]));
const TOP_OPTIONS = [5, 10, 20, 30, 50];
const COLD_START_MS = 3500;
const REQUEST_TIMEOUT_MS = 90000;

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
const main = $('#main');
const form = $('#search-form');
const input = $('#q');
const state = { method: 'semantic', n: 10, q: '', evaluation: null, docs: new Map() };
let inflight = null;

const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) =>
  ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const nf = (digits) => new Intl.NumberFormat('es', { minimumFractionDigits: digits, maximumFractionDigits: digits });
const score4 = nf(4);
const metric4 = nf(4);
const int = new Intl.NumberFormat('es');
const reducedMotion = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

function announce(text) {
  const status = $('#status');
  status.textContent = '';
  requestAnimationFrame(() => { status.textContent = text; });
}

function hashFor(path, params = {}) {
  const query = new URLSearchParams(Object.entries(params).filter(([, v]) => v !== '' && v != null));
  return `#${path}${query.size ? `?${query}` : ''}`;
}

// ---------------------------------------------------------------- API

class ApiError extends Error {
  constructor(code, message) { super(message); this.code = code; }
}

async function api(path, params, signal) {
  const url = `${API}${path}${params ? `?${new URLSearchParams(params)}` : ''}`;
  let response;
  try {
    response = await fetch(url, { signal, headers: { Accept: 'application/json' } });
  } catch (error) {
    if (error.name === 'AbortError') throw error;
    throw new ApiError('network', 'No hay conexión con el servidor de búsqueda. Comprueba tu red e inténtalo de nuevo.');
  }
  const body = await response.json().catch(() => null);
  if (!response.ok) {
    const detail = body?.error;
    if (response.status === 504) throw new ApiError('timeout', 'El servidor tardó demasiado en responder. Vuelve a intentarlo.');
    throw new ApiError(detail?.code ?? 'http', detail?.message ?? `El servidor respondió con el error ${response.status}.`);
  }
  return body;
}

function abortInflight() {
  inflight?.abort();
  inflight = new AbortController();
  return inflight;
}

// ---------------------------------------------------------------- Menú del método (listbox accesible)

const menu = (() => {
  const button = $('#method-button');
  const list = $('#method-list');
  const label = $('#method-label');
  let active = 0;
  let typed = '';
  let typedTimer;

  list.innerHTML = METHODS.map((m) => `
    <li class="method-option" role="option" id="opt-${m.id}" data-id="${m.id}" aria-selected="false">
      <span>${m.label}</span><span class="opt-family">${m.family}</span>
      <svg class="opt-check" viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m5 12.5 4.5 4.5L19 7.5"/></svg>
    </li>`).join('');
  const options = $$('.method-option', list);

  const selectedIndex = () => METHODS.findIndex((m) => m.id === state.method);
  function paint() {
    label.textContent = BY_ID[state.method].label;
    options.forEach((option, index) => option.setAttribute('aria-selected', String(index === selectedIndex())));
  }
  function setActive(index) {
    active = (index + options.length) % options.length;
    options.forEach((option, i) => option.classList.toggle('is-active', i === active));
    list.setAttribute('aria-activedescendant', options[active].id);
    options[active].scrollIntoView({ block: 'nearest' });
  }
  function open() {
    list.hidden = false;
    button.setAttribute('aria-expanded', 'true');
    list.focus();
    setActive(selectedIndex());
  }
  function close(returnFocus = true) {
    if (list.hidden) return;
    list.hidden = true;
    button.setAttribute('aria-expanded', 'false');
    if (returnFocus) button.focus();
  }
  function choose(index) {
    const changed = METHODS[index].id !== state.method;
    state.method = METHODS[index].id;
    paint();
    close();
    if (changed) document.dispatchEvent(new CustomEvent('lente:method'));
  }

  button.addEventListener('click', () => (list.hidden ? open() : close()));
  button.addEventListener('keydown', (event) => {
    if (['ArrowDown', 'ArrowUp'].includes(event.key)) { event.preventDefault(); open(); }
  });
  list.addEventListener('keydown', (event) => {
    const keys = {
      ArrowDown: () => setActive(active + 1), ArrowUp: () => setActive(active - 1),
      Home: () => setActive(0), End: () => setActive(options.length - 1),
      Enter: () => choose(active), ' ': () => choose(active), Escape: () => close(),
    };
    if (keys[event.key]) { event.preventDefault(); keys[event.key](); return; }
    if (event.key === 'Tab') { close(false); return; }
    if (event.key.length === 1 && !event.ctrlKey && !event.metaKey) {   // búsqueda por letra inicial
      typed += event.key.toLowerCase();
      clearTimeout(typedTimer);
      typedTimer = setTimeout(() => { typed = ''; }, 600);
      const match = METHODS.findIndex((m) => m.label.toLowerCase().startsWith(typed));
      if (match >= 0) setActive(match);
    }
  });
  list.addEventListener('mousedown', (event) => event.preventDefault());
  list.addEventListener('click', (event) => {
    const option = event.target.closest('.method-option');
    if (option) choose(options.indexOf(option));
  });
  list.addEventListener('focusout', () => close(false));
  document.addEventListener('pointerdown', (event) => {
    if (!event.target.closest('#method')) close(false);
  });

  paint();
  return { paint };
})();

// ---------------------------------------------------------------- Vistas y enrutado

function setView(view) {
  const apply = () => { document.body.dataset.view = view; };
  if (document.startViewTransition && !reducedMotion() && document.body.dataset.view !== view) {
    document.startViewTransition(apply);
  } else {
    apply();
  }
  $('#nav-eval').toggleAttribute('aria-current', view === 'eval');
  if (view === 'eval') $('#nav-eval').setAttribute('aria-current', 'page');
}

function parseHash() {
  const raw = location.hash.replace(/^#/, '') || '/';
  const [path, query = ''] = raw.split('?');
  return { path, params: new URLSearchParams(query) };
}

async function route({ focusHeading = true } = {}) {
  const { path, params } = parseHash();
  const controller = abortInflight();
  closeDialog();
  window.scrollTo({ top: 0 });
  if (path === '/buscar' || path === '/comparar') {
    const q = (params.get('q') ?? '').trim();
    const method = BY_ID[params.get('m')] ? params.get('m') : state.method;
    const n = TOP_OPTIONS.includes(Number(params.get('n'))) ? Number(params.get('n')) : state.n;
    state.method = method; state.n = n; state.q = q;
    menu.paint();
    input.value = q;
    if (!q) { location.hash = '#/'; return; }
    const compare = path === '/comparar';
    setView(compare ? 'compare' : 'results');
    await showSearch(q, compare, controller, focusHeading);
  } else if (path === '/evaluacion') {
    setView('eval');
    document.title = 'Evaluación FiQA · Lente';
    await showEvaluation(controller, focusHeading);
  } else {
    setView('home');
    document.title = 'Lente · Buscador de FiQA';
    main.replaceChildren();
    if (focusHeading) input.focus({ preventScroll: true });
  }
}

function submitSearch(event) {
  event.preventDefault();
  const q = input.value.replace(/\s+/g, ' ').trim();
  if (!q) {
    showBarMessage('Escribe una consulta para buscar en FiQA.');
    input.focus();
    return;
  }
  showBarMessage('');
  const compare = parseHash().path === '/comparar';
  const target = hashFor(compare ? '/comparar' : '/buscar', { q, m: compare ? '' : state.method, n: state.n });
  if (target === location.hash) route({ focusHeading: false }); else location.hash = target;
}

function showBarMessage(text) {
  let node = $('#bar-msg');
  if (!node) {
    node = document.createElement('p');
    node.id = 'bar-msg'; node.className = 'bar-msg'; node.setAttribute('role', 'alert');
    form.after(node);
  }
  node.textContent = text;
  node.hidden = !text;
  if (text) input.setAttribute('aria-invalid', 'true'); else input.removeAttribute('aria-invalid');
}

form.addEventListener('submit', submitSearch);
input.addEventListener('input', () => showBarMessage(''));
window.addEventListener('hashchange', () => route());
document.addEventListener('lente:method', () => {
  const { path } = parseHash();
  // En /buscar cambia el ranking mostrado; desde la comparación lleva al método elegido.
  if ((path === '/buscar' || path === '/comparar') && state.q) {
    location.hash = hashFor('/buscar', { q: state.q, m: state.method, n: state.n });
  }
});

// ---------------------------------------------------------------- Carga, vacío y error

function renderLoading(message) {
  main.setAttribute('aria-busy', 'true');
  main.innerHTML = `
    <h1 class="view-title" tabindex="-1">${message}</h1>
    <p class="view-sub" id="cold-note" hidden>El servidor de búsqueda se está iniciando. La primera consulta tras un periodo de inactividad tarda unos segundos más; las siguientes son inmediatas.</p>
    <div class="skeleton" aria-hidden="true">
      ${[88, 72, 80, 64, 76].map((w) => `<div><i style="--w:${w}%"></i><i style="--w:${w - 28}%;margin-top:.5rem"></i></div>`).join('')}
    </div>`;
  const timer = setTimeout(() => {
    const note = $('#cold-note');
    if (note) { note.hidden = false; announce('El servidor se está iniciando; la primera búsqueda puede tardar unos segundos.'); }
  }, COLD_START_MS);
  return () => { clearTimeout(timer); main.removeAttribute('aria-busy'); };
}

function renderError(error, retry) {
  main.removeAttribute('aria-busy');
  main.innerHTML = `
    <div class="state error" role="alert">
      <h1 class="view-title" tabindex="-1">No se pudo completar la búsqueda</h1>
      <p>${esc(error.message)}</p>
      <div class="actions">
        <button class="btn" type="button" id="retry">Reintentar</button>
        <a class="btn ghost" href="#/">Volver al inicio</a>
      </div>
    </div>`;
  $('#retry').addEventListener('click', retry);
  announce(`Error: ${error.message}`);
  $('h1', main)?.focus({ preventScroll: true });
}

// Corrección ortográfica opcional que envía el servidor solo cuando un método léxico no encuentra nada.
// La consulta original se queda como está; la corregida solo se ejecuta si la persona pulsa el enlace.
function didYouMean(q, method, suggestion) {
  const fix = typeof suggestion?.query === 'string' ? suggestion.query.trim() : '';
  if (method === 'semantic' || !fix || fix === q) return '';
  return `<a class="btn" id="did-you-mean" href="${hashFor('/buscar', { q: fix, m: method, n: state.n })}">¿Quisiste decir <strong>${esc(fix)}</strong>?</a>`;
}

function renderEmpty(q, method, suggestion) {
  const lexical = method !== 'semantic';
  main.removeAttribute('aria-busy');
  return `
    <div class="state">
      <h2>Sin resultados para “${esc(q)}”</h2>
      <p>${lexical
        ? `${BY_ID[method].label} solo encuentra documentos que comparten palabras con la consulta, y las palabras vacías (the, of, a…) se ignoran.`
        : 'La búsqueda semántica no encontró documentos para esta consulta.'}</p>
      <ul>
        <li>Prueba con otras palabras o con términos más específicos.</li>
        ${lexical ? '<li>Cambia a <strong>Semántico</strong>: compara significados, no solo palabras.</li>' : ''}
        <li>FiQA está en inglés; las consultas en inglés funcionan mejor.</li>
      </ul>
      <div class="actions">
        ${didYouMean(q, method, suggestion)}
        ${lexical ? `<a class="btn" href="${hashFor('/buscar', { q, m: 'semantic', n: state.n })}">Buscar con Semántico</a>` : ''}
        <a class="btn ghost" href="${hashFor('/comparar', { q, n: state.n })}">Comparar los cuatro métodos</a>
      </div>
    </div>`;
}

// ---------------------------------------------------------------- Resultados

function toolbar(compare, q) {
  const single = hashFor('/buscar', { q, m: state.method, n: state.n });
  const multi = hashFor('/comparar', { q, n: state.n });
  return `
    <div class="toolbar">
      <nav class="segmented" aria-label="Modo de resultados">
        <a href="${single}" ${compare ? '' : 'aria-current="true"'}>Un método</a>
        <a href="${multi}" ${compare ? 'aria-current="true"' : ''}>Comparar los cuatro</a>
      </nav>
      <label class="field">Resultados por método
        <select id="top-n">${TOP_OPTIONS.map((n) => `<option value="${n}" ${n === state.n ? 'selected' : ''}>${n}</option>`).join('')}</select>
      </label>
    </div>`;
}

function bindToolbar(compare, q) {
  $('#top-n')?.addEventListener('change', (event) => {
    state.n = Number(event.target.value);
    location.hash = compare ? hashFor('/comparar', { q, n: state.n }) : hashFor('/buscar', { q, m: state.method, n: state.n });
  });
}

function judgedNote(judged) {
  if (!judged) return '';
  return `<p class="note-judged">Esta consulta está en el conjunto de prueba de FiQA (id ${esc(judged.query_id)}): tiene ${judged.relevant_total} documento${judged.relevant_total === 1 ? '' : 's'} relevante${judged.relevant_total === 1 ? '' : 's'} según los juicios de relevancia (qrels) y se marcan como <span class="badge">Relevante</span>.</p>`;
}
//convierte cada resultado en al api en una tarjeta visible para el user
function hitRow(hit, method) {
  state.docs.set(hit.id, hit);
  return `
    <li class="hit">
      <div class="hit-rank" aria-hidden="true">${hit.rank}</div>
      <div class="hit-body">
        <span class="visually-hidden">Posición ${hit.rank}.</span>
        ${hit.title ? `<h3 class="hit-title">${esc(hit.title)}</h3>` : ''}
        <p class="hit-text">${esc(hit.snippet)}</p>
        <div class="hit-meta">
          <span>Documento ${esc(hit.id)}</span>
          <button class="linklike" type="button" data-doc="${esc(hit.id)}">Leer completo</button>
          ${hit.relevant ? '<span class="badge">Relevante</span>' : ''}
        </div>
      </div>
      <div class="hit-score">${score4.format(hit.score)}<small><span aria-hidden="true">puntuación</span><span class="visually-hidden">puntuación propia de ${BY_ID[method].label}</span></small></div>
    </li>`;
}
//usa para pausar hasta que reponda la API y luego renderiza los resultados de la busqueda
async function showSearch(q, compare, controller, focusHeading) {
  const done = renderLoading(compare ? 'Comparando los cuatro métodos…' : `Buscando con ${BY_ID[state.method].label}…`);
  const timeout = setTimeout(() => controller.abort('timeout'), REQUEST_TIMEOUT_MS);
  const retry = () => route({ focusHeading: false });
  try {
    const params = compare ? { q, n: state.n } : { q, method: state.method, n: state.n };
    const data = await api(compare ? '/api/compare' : '/api/search', params, controller.signal); // si es verdadero
    done();//llama a al api de busca o comparacion y guarda la respuesta en data
    if (controller.signal.aborted) return;
    compare ? renderCompare(data) : renderSingle(data);
    document.title = `${q} · ${compare ? 'Comparación' : data.label} · Lente`;
    const heading = $('h1', main);
    if (focusHeading) heading?.focus({ preventScroll: true });
    announce(compare
      ? `${data.distinct} documentos distintos en los cuatro métodos, ${data.in_all_methods} en todos.`
      : `${data.count} resultado${data.count === 1 ? '' : 's'} con ${data.label} en ${Math.round(data.took_ms)} milisegundos.${!data.count && $('#did-you-mean') ? ` ¿Quisiste decir ${data.suggestion.query}?` : ''}`);
  } catch (error) {
    done();
    if (error.name === 'AbortError' && controller.signal.reason !== 'timeout') return;   // sustituida por otra búsqueda
    renderError(error.name === 'AbortError' ? new ApiError('timeout', 'La búsqueda tardó más de lo esperado. Vuelve a intentarlo.') : error, retry);
  } finally {
    clearTimeout(timeout);
  }
}
//musetra lso resultdoas del metodo que eleji en la pantalla
function renderSingle(data) {
  const { q } = state;
  main.innerHTML = `
    <h1 class="view-title" tabindex="-1">Resultados para “${esc(data.query)}”</h1>
    <p class="view-sub">${data.count} con ${data.label} (${BY_ID[data.method].family.toLowerCase()}), calculados en ${Math.round(data.took_ms)} ms. Las puntuaciones solo se pueden comparar dentro del mismo método; para contrastar métodos, usa las posiciones.</p>
    ${toolbar(false, q)}
    ${judgedNote(data.judged)}
    ${data.count ? `<ol class="hits" role="list">${data.hits.map((hit) => hitRow(hit, data.method)).join('')}</ol>` : renderEmpty(data.query, data.method, data.suggestion)}`;
  bindToolbar(false, q);
  bindDocButtons();
}

// ---------------------------------------------------------------- Comparación

let activeColumn = 'semantic';

function rankChips(doc, current) {
  return `<ul class="ranks" aria-label="Posición en cada método">${METHODS.map((m) => {
    const rank = doc.ranks[m.id];
    const cls = [`r-${m.id}`, m.id === current ? 'here' : '', rank ? '' : 'absent'].join(' ');
    const text = rank ? `${m.label}: posición ${rank}` : `${m.label}: fuera de los primeros resultados`;
    return `<li class="${cls}" title="${esc(text)}"><span class="visually-hidden">${esc(text)}</span><span aria-hidden="true">${m.short}<b>${rank ?? '–'}</b></span></li>`;
  }).join('')}</ul>`;
}
//arma las 4 metodos en columas para compararlos
function renderCompare(data) {
  const { q } = state;
  for (const [id, doc] of Object.entries(data.documents)) state.docs.set(id, { ...doc, id });
  const max = Math.max(1, data.n);
  const overlapRows = METHODS.map((a) => `
    <tr><th scope="row"><span class="dot m-${a.id}"></span>${a.label}</th>
      ${METHODS.map((b) => a.id === b.id
        ? '<td class="self" aria-label="el mismo método">–</td>'
        : `<td style="--v:${data.overlap[a.id][b.id] / max}" aria-label="${data.overlap[a.id][b.id]} documentos en común entre ${a.label} y ${b.label}">${data.overlap[a.id][b.id]}</td>`).join('')}
    </tr>`).join('');
  const columns = data.methods.map((m) => `
    <section class="col m-${m.id} ${m.id === activeColumn ? 'is-active' : ''}" id="col-${m.id}" aria-labelledby="h-${m.id}">
      <header class="col-head"><h2 id="h-${m.id}"><span class="dot"></span>${m.label}</h2><p>${m.family}, ${Math.round(m.took_ms)} ms</p></header>
      ${m.hits.length ? `<ol role="list" style="list-style:none;margin:0;padding:0">${m.hits.map((hit) => {
        const doc = data.documents[hit.id];
        return `<li class="chit">
          <div class="chit-rank" aria-hidden="true">${hit.rank}</div>
          <div><span class="visually-hidden">Posición ${hit.rank}.</span>
            ${doc.title ? `<p class="hit-title">${esc(doc.title)}</p>` : ''}
            <p class="chit-text">${esc(doc.snippet)}</p></div>
          <div class="chit-foot">
            ${rankChips(doc, m.id)}
            <button class="linklike" type="button" data-doc="${esc(hit.id)}">Leer</button>
            ${doc.relevant ? '<span class="badge">Relevante</span>' : ''}
          </div></li>`;
      }).join('')}</ol>` : `<div class="state"><p>${m.label} no encontró documentos con palabras en común. Las palabras vacías (the, of, a…) se ignoran.</p></div>`}
    </section>`).join('');

  main.innerHTML = `
    <h1 class="view-title" tabindex="-1">Comparación para “${esc(data.query)}”</h1>
    <p class="view-sub">Los primeros ${data.n} de cada método: ${data.distinct} documentos distintos, ${data.in_all_methods} en los cuatro. Cada documento muestra su posición en los demás métodos; las puntuaciones crudas no se comparan porque cada método usa otra escala.</p>
    ${toolbar(true, q)}
    ${judgedNote(data.judged)}
    <div class="overlap">
      <table><caption class="visually-hidden">Documentos en común entre los primeros ${data.n} resultados de cada par de métodos</caption>
        <thead><tr><th scope="col"><span class="visually-hidden">Método</span></th>${METHODS.map((m) => `<th scope="col">${m.label}</th>`).join('')}</tr></thead>
        <tbody>${overlapRows}</tbody></table>
      <p class="hint">Documentos en común entre los primeros ${data.n} de cada par de métodos.</p>
    </div>
    <div class="tabs segmented" role="tablist" aria-label="Método a mostrar">
      ${METHODS.map((m) => `<button role="tab" type="button" id="tab-${m.id}" data-col="${m.id}" aria-selected="${m.id === activeColumn}" aria-controls="col-${m.id}">${m.label}</button>`).join('')}
    </div>
    <div class="columns">${columns}</div>`;
  bindToolbar(true, q);
  bindDocButtons();
  $$('.tabs [role="tab"]').forEach((tab) => tab.addEventListener('click', () => selectColumn(tab.dataset.col)));
  $('.tabs')?.addEventListener('keydown', (event) => {
    const tabs = $$('.tabs [role="tab"]');
    const index = tabs.findIndex((t) => t.dataset.col === activeColumn);
    const next = { ArrowRight: index + 1, ArrowLeft: index - 1, Home: 0, End: tabs.length - 1 }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    const target = tabs[(next + tabs.length) % tabs.length];
    selectColumn(target.dataset.col);
    target.focus();
  });
  selectColumn(activeColumn);
}

function selectColumn(id) {
  activeColumn = id;
  $$('.tabs [role="tab"]').forEach((tab) => {
    const on = tab.dataset.col === id;
    tab.setAttribute('aria-selected', String(on));
    tab.tabIndex = on ? 0 : -1;
    tab.toggleAttribute('aria-current', on);
  });
  $$('.col').forEach((col) => col.classList.toggle('is-active', col.id === `col-${id}`));
}

// ---------------------------------------------------------------- Documento completo

const dialog = $('#doc-dialog');
let lastTrigger = null;

function bindDocButtons() {
  $$('[data-doc]', main).forEach((button) => button.addEventListener('click', () => openDoc(button.dataset.doc, button)));
}

function openDoc(id, trigger) {
  const doc = state.docs.get(id);
  if (!doc) return;
  lastTrigger = trigger;
  const ranks = doc.ranks
    ? `<p class="hint">Posiciones: ${METHODS.map((m) => `${m.label} ${doc.ranks[m.id] ?? 'fuera del top'}`).join(', ')}.</p>` : '';
  dialog.innerHTML = `
    <div class="dlg-head">
      <div><h2 id="doc-dialog-title">${esc(doc.title || `Documento ${id}`)}</h2><p>${doc.title ? `Documento ${esc(id)} de FiQA` : 'Corpus FiQA'}${doc.relevant ? ', relevante según los qrels' : ''}</p></div>
      <button class="dlg-close" type="button" id="dlg-close">Cerrar</button>
    </div>
    <div class="dlg-body"><p class="hit-text">${esc(doc.text)}</p>${doc.truncated ? '<p class="hint">Texto recortado para la vista previa.</p>' : ''}${ranks}</div>`;
  dialog.showModal();
  $('#dlg-close').addEventListener('click', closeDialog);
}

function closeDialog() {
  if (dialog.open) dialog.close();
}
dialog.addEventListener('click', (event) => { if (event.target === dialog) closeDialog(); });
dialog.addEventListener('close', () => { lastTrigger?.focus?.(); lastTrigger = null; });

// ---------------------------------------------------------------- Evaluación

const evalUi = { metric: 2, sort: { key: 'id', dir: 1 }, filter: '', page: 0, size: 20 };
const METRICS = [['Precision@10', 0], ['Recall@10', 1], ['AP@10', 2]];

async function showEvaluation(controller, focusHeading) {
  const done = renderLoading('Cargando la evaluación…');
  try {
    state.evaluation ??= await api('/api/evaluation', null, controller.signal);
    done();
    if (controller.signal.aborted) return;
    renderEvaluation(state.evaluation);
    if (focusHeading) $('h1', main)?.focus({ preventScroll: true });
    announce('Evaluación de FiQA cargada.');
  } catch (error) {
    done();
    if (error.name === 'AbortError') return;
    renderError(error, () => route({ focusHeading: false }));
  }
}
//toma las metricas que llegaron las metricas que llegaron de la api y la smuestra en la pestanya de evaluacion
function renderEvaluation(ev) {
  const maxMap = Math.max(...ev.methods.map((m) => m.map));
  const rows = ev.methods.map((m) => `
    <tr class="m-${m.id}">
      <th scope="row"><span class="dot"></span>${m.label}<br><small class="hint">${m.family}</small></th>
      <td class="bar-cell"><div class="bar"><span class="bar-track"><span class="bar-fill" style="--p:${(m.map / (maxMap * 1.08)) * 100}%"></span></span><b>${metric4.format(m.map)}</b></div></td>
      <td class="num">${metric4.format(m.mean_precision)}</td>
      <td class="num">${metric4.format(m.mean_recall)}</td>
      <td class="num">${int.format(m.queries_with_hit)} de ${int.format(ev.query_count)}</td>
    </tr>`).join('');
  const h2h = ev.head_to_head.map((p) => {
    const a = BY_ID[p.a], b = BY_ID[p.b], total = p.a_better + p.b_better + p.tie;
    const pct = (v) => `${(v / total) * 100}%`;
    return `<div class="h2h-row" style="--ca:var(--m-${p.a});--cb:var(--m-${p.b})">
      <p><strong>${a.label}</strong> frente a <strong>${b.label}</strong>: ${a.label} es mejor en ${int.format(p.a_better)} consultas, ${b.label} en ${int.format(p.b_better)} y empatan en ${int.format(p.tie)} (${int.format(p.both_zero)} con cero aciertos en ambos).</p>
      <div class="h2h-bar" role="img" aria-label="${a.label} mejor en ${p.a_better}, ${b.label} mejor en ${p.b_better}, empate en ${p.tie}">
        <span class="h2h-a" style="width:${pct(p.a_better)}"></span><span class="h2h-t" style="width:${pct(p.tie)}"></span><span class="h2h-b" style="width:${pct(p.b_better)}"></span>
      </div></div>`;
  }).join('');

  main.innerHTML = `
    <h1 class="view-title" tabindex="-1">Evaluación FiQA</h1>
    <p class="view-sub">${int.format(ev.query_count)} consultas de prueba, corte k=${ev.k}, sobre ${int.format(ev.documents)} documentos y ${int.format(ev.relevant_total)} juicios de relevancia. Valores leídos de <code>${esc(ev.source)}</code>, sin recalcular.</p>

    <section class="eval-section" aria-labelledby="h-summary">
      <h2 id="h-summary">Resumen por método</h2>
      <div class="table-wrap"><table class="data">
        <caption>MAP@${ev.k}: media de la precisión promedio por consulta. Precision y Recall son medias sobre las ${int.format(ev.query_count)} consultas.</caption>
        <thead><tr><th scope="col">Método</th><th scope="col">MAP@${ev.k}</th><th scope="col" class="num">Precision@${ev.k} media</th><th scope="col" class="num">Recall@${ev.k} medio</th><th scope="col" class="num">Consultas con algún acierto</th></tr></thead>
        <tbody>${rows}</tbody></table></div>
    </section>

    <section class="eval-section" aria-labelledby="h-h2h">
      <h2 id="h-h2h">Dónde gana cada método</h2>
      <p class="hint">Se compara el AP@${ev.k} de cada consulta, es decir, la calidad del orden del ranking; no las puntuaciones crudas, que están en escalas distintas.</p>
      <div class="h2h">${h2h}</div>
    </section>

    <section class="eval-section" aria-labelledby="h-queries">
      <h2 id="h-queries">Las ${int.format(ev.query_count)} consultas, una por una</h2>
      <div class="toolbar">
        <div class="segmented" role="group" aria-label="Métrica a mostrar">
          ${METRICS.map(([name, i]) => `<button type="button" data-metric="${i}" aria-pressed="${i === evalUi.metric}">${name}</button>`).join('')}
        </div>
        <label class="field">Filtrar consultas <input type="search" id="eval-filter" value="${esc(evalUi.filter)}" placeholder="Texto de la consulta"></label>
      </div>
      <div class="table-wrap" id="eval-table"></div>
      <div class="pager" id="eval-pager"></div>
    </section>

    <section class="eval-section" aria-labelledby="h-defs">
      <h2 id="h-defs">Cómo leer estas métricas</h2>
      <dl class="defs">
        <dt>Corte k=${ev.k}</dt><dd>Solo se evalúan los ${ev.k} primeros documentos de cada ranking.</dd>
        <dt>Precision@${ev.k}</dt><dd>Documentos relevantes entre los ${ev.k} primeros, dividido entre ${ev.k}. Como cada consulta tiene pocos relevantes (unos ${(ev.relevant_total / ev.query_count).toFixed(1).replace('.', ',')} de media), el máximo posible es bajo: compara métodos entre sí, no contra 1.</dd>
        <dt>Recall@${ev.k}</dt><dd>Relevantes entre los ${ev.k} primeros, dividido entre todos los relevantes de la consulta.</dd>
        <dt>AP@${ev.k} y MAP@${ev.k}</dt><dd>Suma de la precisión en cada posición con acierto dentro del top ${ev.k}, dividida entre el total de relevantes; el MAP es su media sobre las consultas. Premia colocar los relevantes arriba.</dd>
        <dt>Por qué se comparan posiciones</dt><dd>Jaccard va de 0 a 1, el coseno de TF-IDF y el de los embeddings son cosenos en espacios distintos y BM25 no tiene tope: 0,7 en un método no equivale a 0,7 en otro. Las métricas y las vistas de comparación usan el orden del ranking.</dd>
      </dl>
    </section>`;

  $$('[data-metric]', main).forEach((button) => button.addEventListener('click', () => {
    evalUi.metric = Number(button.dataset.metric);
    $$('[data-metric]', main).forEach((b) => b.setAttribute('aria-pressed', String(b === button)));
    paintQueryTable();
  }));
  $('#eval-filter').addEventListener('input', (event) => { evalUi.filter = event.target.value; evalUi.page = 0; paintQueryTable(); });
  paintQueryTable();
}

function paintQueryTable() {
  const ev = state.evaluation;
  const { metric, sort, filter, size } = evalUi;
  const needle = filter.trim().toLowerCase();
  const value = (query, key) => (key === 'id' ? Number(query.id) : key === 'text' ? query.text.toLowerCase() : query[key][metric]);
  const filtered = ev.queries.filter((q) => !needle || q.text.toLowerCase().includes(needle));
  filtered.sort((a, b) => {
    const [x, y] = [value(a, sort.key), value(b, sort.key)];
    return (x < y ? -1 : x > y ? 1 : 0) * sort.dir || Number(a.id) - Number(b.id);
  });
  const pages = Math.max(1, Math.ceil(filtered.length / size));
  evalUi.page = Math.min(evalUi.page, pages - 1);
  const slice = filtered.slice(evalUi.page * size, (evalUi.page + 1) * size);
  const ariaSort = (key) => (sort.key === key ? (sort.dir === 1 ? 'ascending' : 'descending') : 'none');
  const head = (key, text, cls = '') => `<th scope="col" class="${cls}" aria-sort="${ariaSort(key)}"><button type="button" data-sort="${key}">${text}</button></th>`;

  $('#eval-table').innerHTML = filtered.length ? `<table class="data">
    <caption>${METRICS[metric][0]} por consulta. ${int.format(filtered.length)} consulta${filtered.length === 1 ? '' : 's'}${needle ? ' con ese texto' : ''}.</caption>
    <thead><tr>${head('text', 'Consulta')}${METHODS.map((m) => head(m.id, m.label, 'num')).join('')}<th scope="col"><span class="visually-hidden">Acciones</span></th></tr></thead>
    <tbody>${slice.map((q) => `<tr><td class="q-cell">${esc(q.text)}</td>
      ${METHODS.map((m) => `<td class="num">${metric4.format(q[m.id][metric])}</td>`).join('')}
      <td><a href="${hashFor('/comparar', { q: q.text, n: 10 })}" aria-label="Comparar los cuatro métodos para: ${esc(q.text)}">Comparar</a></td></tr>`).join('')}</tbody></table>`
    : '<div class="state"><h2>Ninguna consulta coincide</h2><p>Prueba con otra palabra o borra el filtro.</p></div>';

  $('#eval-pager').innerHTML = filtered.length ? `
    <button type="button" id="pg-prev" ${evalUi.page === 0 ? 'disabled' : ''}>Anterior</button>
    <span>Página ${evalUi.page + 1} de ${pages}</span>
    <button type="button" id="pg-next" ${evalUi.page >= pages - 1 ? 'disabled' : ''}>Siguiente</button>` : '';
  $$('[data-sort]', main).forEach((button) => button.addEventListener('click', () => {
    const key = button.dataset.sort;
    evalUi.sort = { key, dir: sort.key === key ? -sort.dir : key === 'text' ? 1 : -1 };
    paintQueryTable();
    $(`[data-sort="${key}"]`)?.focus();
  }));
  $('#pg-prev')?.addEventListener('click', () => { evalUi.page -= 1; paintQueryTable(); $('#pg-prev')?.focus(); });
  $('#pg-next')?.addEventListener('click', () => { evalUi.page += 1; paintQueryTable(); $('#pg-next')?.focus(); });
}

// ---------------------------------------------------------------- Arranque

// Despierta la API (en Vercel puede estar en frío) mientras la persona escribe, sin forzar una búsqueda.
api('/api/health').then((health) => {
  if (health?.documents) $('#foot-corpus').textContent = `Corpus FiQA · ${int.format(health.documents)} documentos`;
}).catch(() => {});

route({ focusHeading: false });
