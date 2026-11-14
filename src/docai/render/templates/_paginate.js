// Постраничная вёрстка: переносит строки таблицы на новые страницы и собирает bbox полей.
// Единицы bbox — пункты PDF (1 CSS px = 0,75 pt), начало координат — левый верхний угол страницы.
async function docaiLayout() {
  await document.fonts.ready;
  const src = document.getElementById('source');
  const head = src.querySelector('.blk-head');
  const table = src.querySelector('table.items');
  const foot = src.querySelector('.blk-foot');
  const pagesEl = document.getElementById('pages');
  const rows = Array.from(table.tBodies[0].rows);
  const pages = [];

  function newPage() {
    const page = document.createElement('section');
    page.className = 'page';
    const body = document.createElement('div');
    body.className = 'page-body';
    page.appendChild(body);
    pagesEl.appendChild(page);
    pages.push(page);
    return body;
  }
  const overflow = (b) => b.scrollHeight > b.clientHeight + 0.5;
  function newTable() {
    const t = table.cloneNode(false);
    Array.from(table.children).forEach((c) => {
      if (c.tagName !== 'TBODY') t.appendChild(c.cloneNode(true));
    });
    const tb = document.createElement('tbody');
    t.appendChild(tb);
    return { t, tb };
  }

  let body = newPage();
  body.appendChild(head);
  let cur = newTable();
  body.appendChild(cur.t);
  for (const r of rows) {
    cur.tb.appendChild(r);
    if (overflow(body) && cur.tb.rows.length > 1) {
      // строки итогов не оставляем одни на странице: переносим их вместе со строкой данных
      const moved = [r];
      if (r.classList.contains('tot')) {
        let prev = r.previousElementSibling;
        while (prev && prev.classList.contains('tot')) {
          moved.unshift(prev);
          prev = prev.previousElementSibling;
        }
        if (prev && cur.tb.rows.length > moved.length + 1) moved.unshift(prev);
      }
      moved.forEach((m) => cur.tb.removeChild(m));
      body = newPage();
      cur = newTable();
      body.appendChild(cur.t);
      moved.forEach((m) => cur.tb.appendChild(m));
    }
  }
  body.appendChild(foot);
  if (overflow(body)) {
    body.removeChild(foot);
    body = newPage();
    body.appendChild(foot);
  }
  src.remove();

  if (pages.length > 1) {
    pages.forEach((p, i) => {
      const no = document.createElement('div');
      no.className = 'page-no';
      no.textContent = 'Стр. ' + (i + 1) + ' из ' + pages.length;
      p.appendChild(no);
    });
  }

  const PT = 0.75;
  const fields = [];
  const sizes = [];
  pages.forEach((p, i) => {
    const pr = p.getBoundingClientRect();
    sizes.push([+(pr.width * PT).toFixed(2), +(pr.height * PT).toFixed(2)]);
    p.querySelectorAll('[data-field]').forEach((el) => {
      const rects = Array.from(el.getClientRects()).filter((r) => r.width > 0 && r.height > 0);
      if (!rects.length) return;
      const x0 = Math.min(...rects.map((r) => r.left)) - pr.left;
      const y0 = Math.min(...rects.map((r) => r.top)) - pr.top;
      const x1 = Math.max(...rects.map((r) => r.right)) - pr.left;
      const y1 = Math.max(...rects.map((r) => r.bottom)) - pr.top;
      fields.push({
        field: el.dataset.field,
        page: i + 1,
        bbox: [x0, y0, x1, y1].map((v) => +(v * PT).toFixed(2)),
        text: el.textContent.replace(/\s+/g, ' ').trim(),
      });
    });
  });
  return { page_count: pages.length, page_size_pt: sizes[0], fields };
}
