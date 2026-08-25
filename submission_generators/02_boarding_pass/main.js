// --- State Management ---
const state = {
  airlines: [],
  currentAirline: null,
  generated: false
};

// --- Init ---
document.addEventListener('DOMContentLoaded', async () => {
  // Set default date to today
  document.getElementById('flight-date-depart').valueAsDate = new Date();
  document.getElementById('flight-date-arrival').valueAsDate = new Date();

  // Load Airlines
  await loadAirlines();

  // Set initial zoom
  updateZoom(1);
});

// --- Core Functions ---

async function loadAirlines() {
  try {
    const response = await fetch('airlines.json');
    if (!response.ok) throw new Error("Could not load airlines.json");
    const data = await response.json();
    state.airlines = data;
    populateAirlineDropdown(data);
  } catch (error) {
    showAlert("Failed to load airlines list: " + error.message, "error");
  }
}

function populateAirlineDropdown(airlines) {
  const select = document.getElementById('airline-select');
  select.innerHTML = '';
  airlines.forEach(airline => {
    const option = document.createElement('option');
    option.value = airline.id;
    option.textContent = airline.name;
    option.dataset.folder = airline.folder;
    option.dataset.encoder = airline.encoder;
    select.appendChild(option);
  });
}

async function generatePass() {
  if (!validateForm()) return;

  showAlert("Generating pass...", "warning");
  disableControls(true);

  const airlineId = document.getElementById('airline-select').value;
  const airlineOption = document.getElementById('airline-select').selectedOptions[0];
  const folder = airlineOption.dataset.folder;
  const encoder = airlineOption.dataset.encoder;

  // Gather Data
  const formData = {
    title: sanitize(document.getElementById('title').value),
    firstname: sanitize(document.getElementById('first-name').value.toUpperCase()),
    lastname: sanitize(document.getElementById('last-name').value.toUpperCase()),
    from: sanitize(document.getElementById('from').value.toUpperCase()),
    to: sanitize(document.getElementById('to').value.toUpperCase()),
    departDate: formatDate(document.getElementById('flight-date-depart').value),
    departTime: document.getElementById('depart-time').value,
    arrivalDate: formatDate(document.getElementById('flight-date-arrival').value),
    arrivalTime: document.getElementById('arrival-time').value,
    flight: sanitize(document.getElementById('flight-num').value.toUpperCase()),
    gate: sanitize(document.getElementById('gate').value.toUpperCase()),
    seat: sanitize(document.getElementById('seat').value.toUpperCase()),
    pnr: sanitize(document.getElementById('pnr').value.toUpperCase()),
    seq: sanitize(document.getElementById('seq').value),
    zone: document.getElementById('zone').value,
    boardTime: document.getElementById('board-time').value
  };

  try {
    // 1. Fetch Template
    let templateHTML = await fetchTemplate(`airlines/${folder}/template.html`);

    // 2. Fetch CSS (if exists)
    let cssContent = "";
    try {
      cssContent = await fetchTemplate(`airlines/${folder}/style.css`);
    } catch (e) { /* CSS optional or handled inside template */ }

    // 3. Generate Barcode
    const depart_date = new Date(formData.departDate);
    const doy = Math.floor((Date.UTC(depart_date.getFullYear(), depart_date.getMonth(), depart_date.getDate()) - Date.UTC(depart_date.getFullYear(), 0, 0)) / 86400000);
    const bcbpData = `M1${formData.lastname}/${formData.firstname} ${formData.pnr} ${formData.from}${formData.to}${formData.flight} ${doy}Y0${formData.seat}${formData.seq} 100`;

    console.log(bcbpData);

    const barcodeImg = await generateBarcode(bcbpData, encoder);
    formData.barcode = barcodeImg;

    // 4. Inject Data into Template
    let finalHTML = injectData(templateHTML, formData);

    // 5. If CSS was fetched externally, inject it (for cleaner templates)
    if (cssContent && !finalHTML.includes('<style>')) {
      finalHTML = `<style>${cssContent}</style>` + finalHTML;
    }

    // 6. Render to Iframe
    renderToIframe(finalHTML);

    // Enable Buttons
    state.generated = true;
    toggleActionButtons(true);
    showAlert("Boarding pass generated successfully!", "warning"); // Reuse warning style for success color logic if needed, or create success

  } catch (error) {
    console.error(error);
    showAlert("Error: " + error.message + ". Using fallback.", "error");

    // Fallback Logic
    try {
      const universalHTML = await fetchTemplate('airlines/zzz_sample/template.html');
      const universalCSS = await fetchTemplate('airlines/zzz_sample/style.css');
      const bcbpData = `M1${formData.lastname}/${formData.firstname} ${formData.pnr} ${formData.from}${formData.to}${formData.flight} ${doy}Y0${formData.seat}${formData.seq} 100`;
      const barcodeImg = await generateBarcode(bcbpData, encoder);
      formData.barcode = barcodeImg;

      let finalHTML = injectData(universalHTML, formData);
      finalHTML = `<style>${universalCSS}</style>` + finalHTML;
      renderToIframe(finalHTML);

      state.generated = true;
      toggleActionButtons(true);
    } catch (fbError) {
      showAlert("Critical Failure: Could not load fallback template.", "error");
    }
  }

  disableControls(false);
}

// --- Helper Functions ---

async function fetchTemplate(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return await response.text();
}

function injectData(template, data) {
  let html = template;
  for (const [key, value] of Object.entries(data)) {
    // Regex match global case-insensitive for {{key}}
    const regex = new RegExp(`{{${key}}}`, 'gi');
    html = html.replace(regex, value);
  }
  return html;
}

function generateBarcode(text, encoder) {
  return new Promise((resolve, reject) => {
    try {
      const canvas = document.createElement('canvas');
      bwipjs.toCanvas(canvas, {
        bcid: encoder,       // Barcode type
        text: text,    // Text to encode
        scale: 3,               // 3x scaling factor
        // height: 10,              // Bar height, in millimeters
        // includetext: true,            // Show human-readable text
        textxalign: 'center',        // Always good to set this
        backgroundcolor: '#FFFFFF'
      });

      resolve(`<img src="${canvas.toDataURL()}" alt="Barcode" />`);
    } catch (e) {
      console.error("Barcode gen failed", e);
      // Fallback text if barcode fails
      resolve(`<div style="background:#eee; padding:5px; font-size:10px;">BARCODE ERROR</div>`);
    }
  });
}

function renderToIframe(html) {
  const iframe = document.getElementById('preview');
  const doc = iframe.contentWindow.document;
  doc.open();
  doc.write(html);
  doc.close();
}

function sanitize(str) {
  return DOMPurify.sanitize(str, { ALLOWED_TAGS: [] });
}

function validateForm() {
  const form = document.getElementById('bp-form');
  if (!form.checkValidity()) {
    form.reportValidity();
    return false;
  }
  return true;
}

function formatDate(dateString) {
  const options = { day: 'numeric', month: 'short', year: 'numeric' };
  return new Date(dateString).toLocaleDateString('en-US', options).toUpperCase();
}

// --- UI Interactions ---

function updateZoom(val) {
  const wrapper = document.getElementById('preview-wrapper');
  const label = document.getElementById('zoom-val');
  wrapper.style.transform = `scale(${val})`;
  label.textContent = Math.round(val * 100) + '%';
}

function toggleDarkMode() {
  document.body.classList.toggle('dark-mode');
}

function showAlert(msg, type) {
  const el = document.getElementById('status-msg');
  el.textContent = msg;
  el.className = `alert ${type}`;
  el.style.display = 'block';
  setTimeout(() => {
    el.style.display = 'none';
  }, 5000);
}

function toggleActionButtons(enabled) {
  document.getElementById('btn-print').disabled = !enabled;
  document.getElementById('btn-pdf').disabled = !enabled;
}

function disableControls(disabled) {
  const btns = document.querySelectorAll('button');
  btns.forEach(b => {
    if (b.id !== 'btn-print' && b.id !== 'btn-pdf') b.disabled = disabled;
  });
}

function confirmReset() {
  if (confirm("Are you sure you want to reset the form? All unsaved changes will be lost.")) {
    document.getElementById('bp-form').reset();
    document.getElementById('flight-date-depart').valueAsDate = new Date();
    // Clear iframe
    const iframe = document.getElementById('preview');
    const doc = iframe.contentWindow.document;
    doc.open();
    doc.write('');
    doc.close();
    toggleActionButtons(false);
    showAlert("Form reset.", "warning");
  }
}

function printPass() {
  const iframe = document.getElementById('preview');
  iframe.contentWindow.focus();
  iframe.contentWindow.print();
}

function exportPDF() {
  const { jsPDF } = window.jspdf;
  const iframe = document.getElementById('preview');
  const body = iframe.contentDocument.body;

  // Get Form Data for Filename
  const pnr = document.getElementById('pnr').value;
  const airline = document.getElementById('airline-select').value;

  showAlert("Generating PDF...", "warning");

  html2canvas(body, {
    scale: 2, // High resolution
    useCORS: true,
    logging: false
  }).then(canvas => {
    const imgData = canvas.toDataURL('image/png');
    // A4 dimensions: 210mm x 297mm
    const pdf = new jsPDF('p', 'mm', 'a4');
    const pdfWidth = pdf.internal.pageSize.getWidth();
    const pdfHeight = pdf.internal.pageSize.getHeight();

    pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
    pdf.save(`BP_${pnr}_${airline}.pdf`);
    showAlert("PDF Downloaded.", "warning");
  }).catch(err => {
    console.error(err);
    showAlert("PDF Generation failed.", "error");
  });
}
