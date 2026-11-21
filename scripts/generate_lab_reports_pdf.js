// Builds the three A4 reports from their checked Markdown sources.
const fs = require('fs');
const path = require('path');
const os = require('os');

function loadPdfKit() {
  if (process.env.PDFKIT_MODULE) return require(process.env.PDFKIT_MODULE);
  try { return require('pdfkit'); } catch (_) { /* Use the installed Jupyter bundle. */ }
  const extensions = path.join(os.homedir(), '.vscode-server', 'extensions');
  const jupyter = fs.readdirSync(extensions).find(name => name.startsWith('ms-toolsai.jupyter-'));
  if (!jupyter) throw new Error('PDFKit is unavailable; set PDFKIT_MODULE to its entry file.');
  return require(path.join(extensions, jupyter, 'dist/node_modules/pdfkit/js/pdfkit.standalone/index.js'));
}
const PDFDocument = loadPdfKit();

const root = path.resolve(__dirname, '..');
const regular = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';
const bold = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';
const dark = '#1e2c40';
const gray = '#566476';
const line = '#cfd6df';
const WIDTH = 595.276;
const HEIGHT = 841.89;
const LEFT = 59;
const RIGHT = 55;
const BODY_WIDTH = WIDTH - LEFT - RIGHT;
const BOTTOM = 775;

function clean(text) {
  return text
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/&nbsp;/g, ' ')
    .trim();
}

function parseMarkdown(source) {
  const blocks = [];
  const lines = source.split(/\r?\n/);
  let paragraph = [];
  function flush() {
    if (paragraph.length) blocks.push({ type: 'paragraph', text: clean(paragraph.join(' ')) });
    paragraph = [];
  }
  for (let i = 0; i < lines.length; i++) {
    const s = lines[i].trim();
    if (!s) { flush(); continue; }
    if (s.startsWith('# ')) { flush(); continue; }
    if (s.startsWith('## ')) { flush(); blocks.push({ type: 'heading', text: clean(s.slice(3)) }); continue; }
    if (s.startsWith('|')) {
      flush();
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        const cells = lines[i].trim().split('|').slice(1, -1).map(clean);
        if (!cells.every(c => /^:?-+:?$/.test(c))) rows.push(cells);
        i++;
      }
      i--;
      blocks.push({ type: 'table', rows });
      continue;
    }
    paragraph.push(s);
  }
  flush();
  return blocks;
}

function makePdf(number, subtitle, topic) {
  const folder = path.join(root, 'Отчёты', `lab_${number}`);
  const input = path.join(folder, `Отчёт ЛР-${number}.md`);
  const output = path.join(folder, `Отчёт ЛР-${number}.pdf`);
  const blocks = parseMarkdown(fs.readFileSync(input, 'utf8'));
  const doc = new PDFDocument({ size: 'A4', margins: { top: 0, bottom: 0, left: 0, right: 0 }, bufferPages: true, info: {
    Title: `Отчёт по лабораторной работе № ${number}. ${subtitle}`,
    Author: 'Исмагилов Артём Азаматович',
    Subject: 'Adaptive Multi-Agent LLM System',
  } });
  doc.registerFont('Regular', fs.readFileSync(regular)).registerFont('Bold', fs.readFileSync(bold));
  const stream = fs.createWriteStream(output);
  doc.pipe(stream);

  // The cover follows the placement and wording of the approved Lab 1 report.
  doc.font('Bold').fillColor(dark).fontSize(19).text(`ОТЧЁТ\nПО ЛАБОРАТОРНОЙ РАБОТЕ № ${number}`,
    LEFT, 264, { width: BODY_WIDTH, align: 'center', lineGap: 5 });
  doc.font('Regular').fontSize(12).text(`«${subtitle}»`, LEFT + 40, 370,
    { width: BODY_WIDTH - 80, align: 'center', lineGap: 5 });
  doc.fontSize(10.5).text(`Тема: ${topic}`, LEFT + 55, 459,
    { width: BODY_WIDTH - 110, align: 'center', lineGap: 5 });
  doc.fontSize(10).text('Выполнил:', 76, 615);
  doc.font('Bold').text('Исмагилов Артём Азаматович', 178, 615);
  doc.moveTo(178, 634).lineTo(502, 634).lineWidth(0.5).strokeColor(dark).stroke();
  doc.font('Regular').text('Группа:', 76, 660);
  doc.font('Bold').text('Б24-517', 178, 660);
  doc.moveTo(178, 679).lineTo(502, 679).stroke();
  doc.font('Bold').fontSize(11).text('2026', LEFT, 764, { width: BODY_WIDTH, align: 'center' });

  doc.addPage(); // Contents, populated once page numbers are known.
  doc.addPage();
  let y = 78;
  let section = 0;
  let table = 0;
  const toc = [];
  function newBodyPage() { doc.addPage(); y = 78; }
  function ensure(height) { if (y + height > BOTTOM) newBodyPage(); }
  function drawParagraph(text, options = {}) {
    const size = options.size || 10;
    const x = options.x || LEFT;
    const width = options.width || BODY_WIDTH;
    const font = options.bold ? 'Bold' : 'Regular';
    const gap = options.gap === undefined ? 3 : options.gap;
    doc.font(font).fontSize(size);
    const h = doc.heightOfString(text, { width, lineGap: gap, align: options.align || 'justify' });
    ensure(h + (options.after || 11));
    doc.fillColor(options.color || dark).text(text, x, y, { width, lineGap: gap, align: options.align || 'justify' });
    y = doc.y + (options.after || 11);
  }
  function drawTable(rows) {
    table++;
    ensure(30);
    drawParagraph(`Таблица ${table} - результаты сравнения`, { bold: true, size: 9.5, after: 8, align: 'left' });
    const headers = rows[0];
    for (const [index, row] of rows.slice(1).entries()) {
      const label = row.slice(0, headers.length === 3 ? 1 : 2).join(' - ');
      const offset = headers.length === 3 ? 1 : 2;
      const details = row.slice(offset).map((v, j) => `${headers[j + offset]}: ${v}`).join('; ');
      doc.font('Bold').fontSize(9.4);
      const lh = doc.heightOfString(label, { width: BODY_WIDTH - 24, lineGap: 2 });
      doc.font('Regular').fontSize(9.1);
      const dh = doc.heightOfString(details, { width: BODY_WIDTH - 24, lineGap: 3 });
      const block = lh + dh + 15;
      ensure(block);
      if (index % 2 === 0) doc.rect(LEFT, y - 3, BODY_WIDTH, block).fill('#f5f7fa');
      doc.fillColor(dark).font('Bold').fontSize(9.4).text(label, LEFT + 12, y, { width: BODY_WIDTH - 24, lineGap: 2 });
      y = doc.y + 3;
      doc.font('Regular').fontSize(9.1).fillColor(gray).text(details, LEFT + 12, y,
        { width: BODY_WIDTH - 24, lineGap: 3 });
      y = doc.y + 9;
    }
    y += 6;
  }

  for (const [blockIndex, block] of blocks.entries()) {
    if (block.type === 'heading') {
      section++;
      const following = blocks[blockIndex + 1];
      let nextHeight = 0;
      if (following && following.type === 'paragraph') {
        doc.font('Regular').fontSize(10);
        nextHeight = doc.heightOfString(following.text, { width: BODY_WIDTH, lineGap: 3 });
      }
      ensure(38 + Math.min(nextHeight, 180));
      const title = `${section}. ${block.text}`;
      toc.push({ title, page: doc.bufferedPageRange().start + doc.bufferedPageRange().count });
      doc.fillColor(dark).font('Bold').fontSize(13).text(title, LEFT, y, { width: BODY_WIDTH });
      y = doc.y + 14;
    } else if (block.type === 'paragraph') drawParagraph(block.text);
    else if (block.type === 'table') drawTable(block.rows);
  }

  const pageCount = doc.bufferedPageRange().count;
  doc.switchToPage(1);
  doc.fillColor(dark).font('Bold').fontSize(17).text('Содержание', LEFT, 82, { width: BODY_WIDTH });
  let ty = 138;
  for (const item of toc) {
    doc.font('Regular').fillColor(dark).fontSize(10.5).text(item.title, LEFT, ty,
      { width: BODY_WIDTH - 50 });
    doc.text(String(item.page), WIDTH - RIGHT - 35, ty, { width: 35, align: 'right' });
    ty += Math.max(28, doc.heightOfString(item.title, { width: BODY_WIDTH - 50 }) + 11);
    doc.moveTo(LEFT, ty - 7).lineTo(WIDTH - RIGHT, ty - 7).lineWidth(0.3).strokeColor(line).stroke();
  }
  for (let index = 1; index < pageCount; index++) {
    doc.switchToPage(index);
    doc.moveTo(LEFT, 795).lineTo(WIDTH - RIGHT, 795).lineWidth(0.35).strokeColor(line).stroke();
    doc.fillColor(gray).font('Regular').fontSize(8)
      .text(`Лабораторная работа № ${number} - Adaptive Multi-Agent LLM System`, LEFT, 806,
        { width: BODY_WIDTH - 30 });
    doc.text(String(index + 1), WIDTH - RIGHT - 25, 806, { width: 25, align: 'right' });
  }
  doc.end();
  return new Promise((resolve, reject) => { stream.on('finish', () => resolve({ output, pages: pageCount })); stream.on('error', reject); });
}

(async () => {
  const jobs = [
    [2, 'Распределение задач между агентами', 'Adaptive Multi-Agent LLM System - сравнение механизмов назначения задач'],
    [3, 'Переговоры и разрешение конфликтов', 'Adaptive Multi-Agent LLM System - распределение ограниченного ресурса'],
    [4, 'Надёжность и безопасность многоагентной системы', 'Adaptive Multi-Agent LLM System - проверка защитных механизмов'],
  ];
  for (const job of jobs) console.log(await makePdf(...job));
})().catch(error => { console.error(error); process.exitCode = 1; });
