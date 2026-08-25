// ==========================================
// Flight E-Ticket Generator
// Renders an EaseMyTrip style e-ticket: any number of passengers, any number
// of journeys (onward / return, each one or more flight segments), fare
// summary and cancellation charges - all on a single printable sheet.
//
// Contract with template.html: the {{TOKENS}} below are the only slots the
// template exposes. Repeated markup (leg cards, traveller tables) is built
// here because token substitution cannot express repetition.
// ==========================================

// --- State Management ---
const state = {
  booking: { id: '', bookedOn: '' },
  passengers: [],
  journeys: [],
  fare: { basic: 0, taxes: 0, discount: 0, emtFee: 300, fees: [] },
  cancellation: []
};

let airports = [];
let airlines = [];
let templateHTML = '';
let idCounter = 0;

// --- Init ---
document.addEventListener('DOMContentLoaded', async () => {
  await loadData();
  seedDefaults();
  renderAll();
  updateZoom(1);
});

async function loadData() {
  try {
    [airports, airlines, templateHTML] = await Promise.all([
      fetchJSON('airports.json'),
      fetchJSON('airlines.json'),
      fetchText('template.html')
    ]);
  } catch (error) {
    showAlert("Failed to load generator data: " + error.message, "error");
  }
}

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return response.json();
}

async function fetchText(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}`);
  return response.text();
}

// ==========================================
// DEFAULTS
// ==========================================

const DEFAULT_PAX = [
  { title: 'Mrs', first: 'TULU', last: 'SEAL' },
  { title: 'Mr', first: 'Dhanonjay Kumar', last: 'SEAL' }
];

function seedDefaults() {
  state.booking.id = generateBookingId();
  state.booking.bookedOn = toLocalInput(new Date());

  state.passengers = DEFAULT_PAX.map(p => ({
    title: p.title, first: p.first, last: p.last,
    seat: '-', meal: '-', baggage: '-',
    insurance: 'Not Confirmed', status: 'Confirm'
  }));

  state.journeys = [
    newJourney('Onward Flight', leg({
      airlineId: 'spicejet', flightNo: '979', operatedBy: 'SG',
      from: { code: 'CCU', terminal: '2' }, depart: { date: offsetDate(7), time: '06:30' },
      to: { code: 'BLR', terminal: '1' }, arrive: { date: offsetDate(7), time: '09:20' },
      cabin: '7Kg', checkin: '15Kg', cabinClass: 'Economy', fareType: 'SpiceSaver Fare'
    })),
    newJourney('Return Flight', leg({
      airlineId: 'indigo', flightNo: '227', operatedBy: '6E',
      from: { code: 'BLR', terminal: '1' }, depart: { date: offsetDate(17), time: '05:00' },
      to: { code: 'CCU', terminal: '' }, arrive: { date: offsetDate(17), time: '07:40' },
      cabin: '7Kg', checkin: '15Kg', cabinClass: 'Economy', fareType: 'Saver Fare'
    }))
  ];

  state.fare = { basic: 19640, taxes: 5660, discount: 1000, emtFee: 300, fees: [{ label: 'K3', amount: 1010 }] };

  state.cancellation = [
    { journeyId: state.journeys[0].id, route: '', type: 'ADT', window: 'Before 96 hours of flight Departure', amount: 3200, auto: true },
    { journeyId: state.journeys[0].id, route: '', type: 'ADT', window: '96 hours to 4 hours of flight Departure', amount: 4275, auto: true },
    { journeyId: state.journeys[1].id, route: '', type: 'ADT', window: '', amount: 0, auto: true }
  ];
}

function newJourney(label, firstLeg) {
  return { id: 'j' + (++idCounter), label, pnr: generatePnr(new Set()), legs: [firstLeg] };
}

function leg(values) {
  return Object.assign({
    airlineId: 'indigo', flightNo: '', operatedBy: '',
    from: { code: '', terminal: '' }, depart: { date: '', time: '' },
    to: { code: '', terminal: '' }, arrive: { date: '', time: '' },
    cabin: '', checkin: '', cabinClass: '', fareType: ''
  }, values);
}

function offsetDate(days) {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return toDateInput(d);
}

function toDateInput(d) {
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

function toLocalInput(d) {
  return `${toDateInput(d)}T${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

// ==========================================
// UNIQUE IDENTIFIERS
// ==========================================

const ISSUED_KEY = 'emt.issued.booking.ids';
const PNR_CHARS = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789';

function loadIssuedIds() {
  try {
    return new Set(JSON.parse(localStorage.getItem(ISSUED_KEY) || '[]'));
  } catch (e) {
    return new Set();
  }
}

function saveIssuedIds(set) {
  try {
    localStorage.setItem(ISSUED_KEY, JSON.stringify([...set]));
  } catch (e) { /* private mode: the id is still unique for this session */ }
}

/* EMT + 9 digits, never reused on this browser. */
function generateBookingId() {
  const issued = loadIssuedIds();
  let id;
  do {
    id = 'EMT' + String(Math.floor(Math.random() * 1e9)).padStart(9, '0');
  } while (issued.has(id));
  issued.add(id);
  saveIssuedIds(issued);
  return id;
}

/* 6 characters, no ambiguous ones (no O/0/I/1), unique across the ticket. */
function generatePnr(exclude) {
  let pnr;
  do {
    pnr = '';
    for (let i = 0; i < 6; i++) {
      pnr += PNR_CHARS[Math.floor(Math.random() * PNR_CHARS.length)];
    }
  } while (exclude.has(pnr));
  return pnr;
}

// ==========================================
// PANEL RENDERING
// ==========================================

function renderAll() {
  syncStaticInputs();
  renderPassengers();
  renderJourneys();
  renderFeeRows();
  renderCancelRows();
}

/* Static inputs live in the HTML, so their values are pushed into the state
   and back out whenever the state is rebuilt. */
function syncStaticInputs() {
  document.getElementById('booking-id').value = state.booking.id;
  document.getElementById('booked-on').value = state.booking.bookedOn;
  document.getElementById('fare-basic').value = state.fare.basic;
  document.getElementById('fare-taxes').value = state.fare.taxes;
  document.getElementById('fare-discount').value = state.fare.discount;
  document.getElementById('fare-emt').value = state.fare.emtFee;
}

function renderPassengers() {
  const html = state.passengers.map((p, i) => `
    <div class="block">
      <div class="block-head"><span>Passenger ${i + 1}</span>
        <button type="button" class="block-remove" data-action="remove-passenger" data-index="${i}" title="Remove passenger">&times;</button>
      </div>
      <div class="row">
        <div class="form-group">
          <label>Title</label>
          <select data-path="passengers.${i}.title">
            ${['Mr', 'Mrs', 'Ms', 'Mstr'].map(t => `<option value="${t}"${p.title === t ? ' selected' : ''}>${t}</option>`).join('')}
          </select>
        </div>
        <div class="form-group" style="flex:2">
          <label>First Name</label>
          <input type="text" data-path="passengers.${i}.first" value="${escAttr(p.first)}" placeholder="TULU">
        </div>
        <div class="form-group" style="flex:2">
          <label>Last Name</label>
          <input type="text" data-path="passengers.${i}.last" value="${escAttr(p.last)}" placeholder="SEAL">
        </div>
      </div>
      <div class="row">
        <div class="form-group"><label>Seat</label><input type="text" data-path="passengers.${i}.seat" value="${escAttr(p.seat)}" placeholder="-"></div>
        <div class="form-group"><label>Meal</label><input type="text" data-path="passengers.${i}.meal" value="${escAttr(p.meal)}" placeholder="-"></div>
        <div class="form-group"><label>Baggage</label><input type="text" data-path="passengers.${i}.baggage" value="${escAttr(p.baggage)}" placeholder="-"></div>
      </div>
      <div class="row">
        <div class="form-group">
          <label>Insurance</label>
          <select data-path="passengers.${i}.insurance">
            ${['Not Confirmed', 'Confirmed'].map(v => `<option value="${v}"${p.insurance === v ? ' selected' : ''}>${v}</option>`).join('')}
          </select>
        </div>
        <div class="form-group">
          <label>Status</label>
          <select data-path="passengers.${i}.status">
            ${['Confirm', 'On Hold', 'Cancelled'].map(v => `<option value="${v}"${p.status === v ? ' selected' : ''}>${v}</option>`).join('')}
          </select>
        </div>
      </div>
    </div>`).join('');
  document.getElementById('passengers-list').innerHTML = html;
}

function renderJourneys() {
  const html = state.journeys.map((j, ji) => `
    <div class="block">
      <div class="block-head"><span>Journey ${ji + 1}</span>
        <button type="button" class="block-remove" data-action="remove-journey" data-index="${ji}" title="Remove journey">&times;</button>
      </div>
      <div class="row">
        <div class="form-group" style="flex:2">
          <label>Section Title</label>
          <input type="text" data-path="journeys.${ji}.label" value="${escAttr(j.label)}" placeholder="Onward Flight">
        </div>
        <div class="form-group">
          <label>PNR</label>
          <input type="text" data-path="journeys.${ji}.pnr" value="${escAttr(j.pnr)}" maxlength="6">
        </div>
        <div class="form-group" style="flex:0 0 auto; display:flex; align-items:flex-end;">
          <button type="button" class="secondary" data-action="regen-pnr" data-index="${ji}" title="New PNR">&#8635;</button>
        </div>
      </div>
      ${j.legs.map((l, li) => renderLegBlock(j, ji, l, li)).join('')}
      <button type="button" class="secondary full-width add-block" data-action="add-leg" data-index="${ji}">＋ Add Stop / Leg</button>
    </div>`).join('');
  document.getElementById('journeys-list').innerHTML = html;
}

function renderLegBlock(journey, ji, l, li) {
  const route = legRouteLabel(l);
  return `
    <div class="block block--leg">
      <div class="block-head"><span>Leg ${li + 1}${route ? ' — ' + esc(route) : ''}</span>
        <button type="button" class="block-remove" data-action="remove-leg" data-index="${ji}" data-leg="${li}" title="Remove leg">&times;</button>
      </div>
      <div class="row">
        <div class="form-group" style="flex:2">
          <label>Airline</label>
          <select data-path="journeys.${ji}.legs.${li}.airlineId">
            ${airlines.map(a => `<option value="${a.id}"${l.airlineId === a.id ? ' selected' : ''}>${esc(a.name)}</option>`).join('')}
          </select>
        </div>
        <div class="form-group"><label>Flight No.</label><input type="text" data-path="journeys.${ji}.legs.${li}.flightNo" value="${escAttr(l.flightNo)}" placeholder="979"></div>
        <div class="form-group"><label>Operated By</label><input type="text" data-path="journeys.${ji}.legs.${li}.operatedBy" value="${escAttr(l.operatedBy)}" placeholder="SG"></div>
      </div>
      <div class="row">
        <div class="form-group" style="flex:2">
          <label>From</label>
          ${airportSelect(`journeys.${ji}.legs.${li}.from.code`, l.from.code)}
        </div>
        <div class="form-group"><label>Terminal</label><input type="text" data-path="journeys.${ji}.legs.${li}.from.terminal" value="${escAttr(l.from.terminal)}" placeholder="2"></div>
      </div>
      <div class="row">
        <div class="form-group"><label>Departure Date</label><input type="date" data-path="journeys.${ji}.legs.${li}.depart.date" value="${escAttr(l.depart.date)}"></div>
        <div class="form-group"><label>Departure Time</label><input type="time" data-path="journeys.${ji}.legs.${li}.depart.time" value="${escAttr(l.depart.time)}"></div>
      </div>
      <div class="row">
        <div class="form-group" style="flex:2">
          <label>To</label>
          ${airportSelect(`journeys.${ji}.legs.${li}.to.code`, l.to.code)}
        </div>
        <div class="form-group"><label>Terminal</label><input type="text" data-path="journeys.${ji}.legs.${li}.to.terminal" value="${escAttr(l.to.terminal)}" placeholder="1"></div>
      </div>
      <div class="row">
        <div class="form-group"><label>Arrival Date</label><input type="date" data-path="journeys.${ji}.legs.${li}.arrive.date" value="${escAttr(l.arrive.date)}"></div>
        <div class="form-group"><label>Arrival Time</label><input type="time" data-path="journeys.${ji}.legs.${li}.arrive.time" value="${escAttr(l.arrive.time)}"></div>
      </div>
      <div class="row">
        <div class="form-group"><label>Cabin Baggage</label><input type="text" data-path="journeys.${ji}.legs.${li}.cabin" value="${escAttr(l.cabin)}" placeholder="7Kg"></div>
        <div class="form-group"><label>Check-in Baggage</label><input type="text" data-path="journeys.${ji}.legs.${li}.checkin" value="${escAttr(l.checkin)}" placeholder="15Kg"></div>
      </div>
      <div class="row">
        <div class="form-group"><label>Class</label><input type="text" data-path="journeys.${ji}.legs.${li}.cabinClass" value="${escAttr(l.cabinClass)}" placeholder="Economy"></div>
        <div class="form-group"><label>Fare Type</label><input type="text" data-path="journeys.${ji}.legs.${li}.fareType" value="${escAttr(l.fareType)}" placeholder="Saver Fare"></div>
      </div>
    </div>`;
}

function airportSelect(path, code) {
  const blank = code ? '' : '<option value="" selected>— select —</option>';
  const options = airports.map(a =>
    `<option value="${a.iata}"${a.iata === code ? ' selected' : ''}>${a.iata} — ${esc(a.city)}</option>`).join('');
  return `<select data-path="${path}">${blank}${options}</select>`;
}

function renderFeeRows() {
  const html = state.fare.fees.map((f, i) => `
    <div class="row">
      <div class="form-group" style="flex:2">
        <label>Fee / surcharge</label>
        <input type="text" data-path="fare.fees.${i}.label" value="${escAttr(f.label)}" placeholder="K3">
      </div>
      <div class="form-group">
        <label>Amount (&#8377;)</label>
        <input type="number" data-path="fare.fees.${i}.amount" value="${f.amount}" min="0" step="1">
      </div>
      <div class="form-group" style="flex:0 0 auto; display:flex; align-items:flex-end;">
        <button type="button" class="block-remove" data-action="remove-fee" data-index="${i}" title="Remove fee row">&times;</button>
      </div>
    </div>`).join('');
  document.getElementById('fee-rows').innerHTML = html;
}

function renderCancelRows() {
  const html = state.cancellation.map((c, i) => `
    <div class="row">
      <div class="form-group" style="flex:2">
        <label>Route</label>
        <input type="text" data-path="cancellation.${i}.route" value="${escAttr(c.auto ? routeLabel(c) : c.route)}" placeholder="Kolkata-Bengaluru">
      </div>
      <div class="form-group">
        <label>Type</label>
        <input type="text" data-path="cancellation.${i}.type" value="${escAttr(c.type)}" placeholder="ADT">
      </div>
    </div>
    <div class="row">
      <div class="form-group" style="flex:2">
        <label>Cancellation Time</label>
        <input type="text" data-path="cancellation.${i}.window" value="${escAttr(c.window)}" placeholder="Before 96 hours of flight Departure">
      </div>
      <div class="form-group">
        <label>Charge (&#8377;)</label>
        <input type="number" data-path="cancellation.${i}.amount" value="${c.amount}" min="0" step="1">
      </div>
      <div class="form-group" style="flex:0 0 auto; display:flex; align-items:flex-end;">
        <button type="button" class="block-remove" data-action="remove-cancel" data-index="${i}" title="Remove charge row">&times;</button>
      </div>
    </div>`).join('');
  document.getElementById('cancel-rows').innerHTML = html;
}

// ==========================================
// PANEL EVENTS
// ==========================================

document.getElementById('controls-panel').addEventListener('input', onFieldChange);
document.getElementById('controls-panel').addEventListener('change', onFieldChange);
document.getElementById('controls-panel').addEventListener('click', onPanelClick);

function onFieldChange(event) {
  const target = event.target;
  const path = target.dataset.path;
  if (!path) return;

  const value = target.type === 'number' ? (target.value === '' ? 0 : Number(target.value)) : target.value;
  const previousAirlineId = path.endsWith('.airlineId') ? getPath(state, path) : null;

  setPath(state, path, value);

  /* An airline switch carries its own operating code; follow it unless the
     user has typed something else. */
  if (previousAirlineId) {
    const airline = airlines.find(a => a.id === value);
    const previous = airlines.find(a => a.id === previousAirlineId);
    const operatedByPath = path.replace(/\.airlineId$/, '.operatedBy');
    const current = getPath(state, operatedByPath);
    if (airline && (!current || current === (previous ? previous.prefix : ''))) {
      setPath(state, operatedByPath, airline.prefix);
      renderJourneys();
    }
  }

  /* A hand typed route wins over the auto generated one. */
  if (path.endsWith('.route')) {
    const index = Number(path.split('.')[1]);
    if (state.cancellation[index]) state.cancellation[index].auto = false;
  }
}

function onPanelClick(event) {
  const button = event.target.closest('[data-action]');
  if (!button) return;
  const index = Number(button.dataset.index);

  switch (button.dataset.action) {
    case 'add-passenger':
      state.passengers.push({
        title: 'Mr', first: '', last: '',
        seat: '-', meal: '-', baggage: '-',
        insurance: 'Not Confirmed', status: 'Confirm'
      });
      renderPassengers();
      break;

    case 'remove-passenger':
      if (state.passengers.length > 1) state.passengers.splice(index, 1);
      renderPassengers();
      break;

    case 'add-journey': {
      const label = state.journeys.length === 0 ? 'Onward Flight'
        : state.journeys.length === 1 ? 'Return Flight'
        : `Journey ${state.journeys.length + 1}`;
      const journey = newJourney(label, leg({
        airlineId: 'indigo', flightNo: '', operatedBy: '6E',
        from: { code: 'BLR', terminal: '' }, depart: { date: offsetDate(17), time: '05:00' },
        to: { code: 'CCU', terminal: '' }, arrive: { date: offsetDate(17), time: '07:40' },
        cabin: '7Kg', checkin: '15Kg', cabinClass: 'Economy', fareType: 'Saver Fare'
      }));
      state.journeys.push(journey);
      state.cancellation.push({
        journeyId: journey.id, route: '', type: 'ADT',
        window: 'Before 96 hours of flight Departure', amount: 0, auto: true
      });
      renderJourneys();
      renderCancelRows();
      break;
    }

    case 'remove-journey': {
      const journey = state.journeys[index];
      if (journey && state.journeys.length > 1) {
        state.journeys.splice(index, 1);
        state.cancellation = state.cancellation.filter(c => c.journeyId !== journey.id);
        renderJourneys();
        renderCancelRows();
      }
      break;
    }

    case 'add-leg': {
      const journey = state.journeys[index];
      const previous = journey.legs[journey.legs.length - 1] || {};
      journey.legs.push(leg({
        airlineId: previous.airlineId || 'indigo',
        flightNo: '', operatedBy: previous.operatedBy || '',
        from: { code: previous.to ? previous.to.code : '', terminal: '' },
        depart: { date: previous.arrive ? previous.arrive.date : offsetDate(17), time: previous.arrive ? previous.arrive.time : '' },
        to: { code: '', terminal: '' },
        arrive: { date: previous.arrive ? previous.arrive.date : offsetDate(17), time: '' },
        cabin: previous.cabin || '7Kg', checkin: previous.checkin || '15Kg',
        cabinClass: previous.cabinClass || 'Economy', fareType: previous.fareType || 'Saver Fare'
      }));
      renderJourneys();
      break;
    }

    case 'remove-leg':
      if (state.journeys[index] && state.journeys[index].legs.length > 1) {
        state.journeys[index].legs.splice(Number(button.dataset.leg), 1);
        renderJourneys();
      }
      break;

    case 'regen-pnr': {
      const journey = state.journeys[index];
      const others = new Set(state.journeys.filter((_, i) => i !== index).map(j => j.pnr));
      journey.pnr = generatePnr(others);
      renderJourneys();
      break;
    }

    case 'add-fee':
      state.fare.fees.push({ label: '', amount: 0 });
      renderFeeRows();
      break;

    case 'remove-fee':
      state.fare.fees.splice(index, 1);
      renderFeeRows();
      break;

    case 'add-cancel':
      state.cancellation.push({
        journeyId: '', route: '', type: 'ADT',
        window: 'Before 96 hours of flight Departure', amount: 0, auto: false
      });
      renderCancelRows();
      break;

    case 'remove-cancel':
      state.cancellation.splice(index, 1);
      renderCancelRows();
      break;
  }
}

// ==========================================
// TICKET RENDERING
// ==========================================

async function generateTicket() {
  if (!validateForm()) return;
  if (!templateHTML) {
    showAlert("Template not loaded - serve this folder over http:// (e.g. Live Server).", "error");
    return;
  }

  showAlert("Generating ticket...", "warning");
  disableControls(true);

  try {
    const html = buildTicket(templateHTML);
    renderToIframe(html);
    document.getElementById('btn-print').disabled = false;
    showAlert("E-ticket generated successfully!", "warning");
  } catch (error) {
    console.error(error);
    showAlert("Error: " + error.message, "error");
  }

  disableControls(false);
}

function buildTicket(template) {
  const tokens = {
    TITLE: `E-Ticket - Booking ID ${state.booking.id} | EaseMyTrip`,
    BOOKING_ID: state.booking.id,
    BOOKED_ON: formatBookedOn(state.booking.bookedOn),
    JOURNEYS: buildJourneys(),
    FARE_INTRO: `Thank you for making the payment of &#8377; ${fareTotal()} . Fare detail are mentioned below`,
    FARE_BASIC: amount(state.fare.basic),
    FARE_TAXES: amount(state.fare.taxes),
    FARE_FEE_ROWS: buildFeeRows(),
    FARE_DISCOUNT: amount(state.fare.discount),
    FARE_TOTAL: amount(fareTotal()),
    EMT_FEE: amount(state.fare.emtFee),
    CANCEL_TABLES: buildCancelTables(),
    AIRLINE_CONTACTS: buildAirlineContacts()
  };

  let html = template;
  for (const [key, value] of Object.entries(tokens)) {
    html = html.split(`{{${key}}}`).join(value);
  }
  return html;
}

function buildJourneys() {
  const coupons = couponNumbers();
  return state.journeys.map(journey => {
    let html = `<section class="flight">\n  <h2 class="section-title">${esc(journey.label)}</h2>`;
    journey.legs.forEach((l, li) => {
      html += '\n  ' + buildLegCard(l, journey, journey === state.journeys[0] && li === 0);
    });
    html += '\n</section>\n';
    html += buildTravellerTable(journey, coupons);
    return html;
  }).join('\n');
}

/* Coupon numbers run across the whole ticket, in document order: the first
   leg of the onward journey is coupon 1, the next leg 2, and so on - the
   ticket number is PNR / passenger / coupon. */
function couponNumbers() {
  const map = new Map();
  let coupon = 0;
  state.journeys.forEach(j => j.legs.forEach(l => map.set(l, ++coupon)));
  return map;
}

function buildLegCard(l, journey, roomy) {
  const airline = airlines.find(a => a.id === l.airlineId) || { name: '', prefix: '', logo: '' };
  const from = airportByCode(l.from.code);
  const to = airportByCode(l.to.code);
  const logo = airline.logo
    ? `<img class="airline-logo" src="${airline.logo}" alt="${escAttr(airline.name)}">`
    : `<div class="airline-badge">${esc(airline.prefix)}</div>`;

  return `<div class="leg${roomy ? ' leg--roomy' : ''}">
    <div class="leg-head">
      <div class="leg-route">
        <span class="city">${esc(from.city)}</span>
        <img class="plane" src="assets/plane.png" alt="">
        <span class="city">${esc(to.city)}</span>
        <p class="leg-date">${longDate(l.depart.date)}</p>
      </div>
      <p class="leg-pnr">PNR: <b>${esc(journey.pnr)}</b></p>
    </div>
    <div class="leg-body">
      <div class="airline">
        ${logo}
        <div class="airline-text">
          <p class="airline-name">${esc(airline.name)}</p>
          <p class="airline-no">${esc(airline.prefix)}- ${esc(l.flightNo)} <span class="muted">(Operated By: ${esc(l.operatedBy)})</span></p>
        </div>
      </div>
      <div class="timeline">
        <div class="stop stop-dep">
          <p class="stop-date">${shortDate(l.depart.date)}</p>
          <p class="stop-time">${esc(l.from.code)} ${esc(l.depart.time)}</p>
          <p class="stop-airport">${airportLine(from, l.from.terminal, true)}</p>
        </div>
        <div class="duration">
          <p class="duration-text">${durationText(l)}</p>
          <img class="duration-line" src="assets/duration.png" alt="">
        </div>
        <div class="stop stop-arr">
          <p class="stop-date">${arrivalDate(l.arrive.date)}</p>
          <p class="stop-time">${esc(l.to.code)} ${esc(l.arrive.time)}</p>
          <p class="stop-airport">${airportLine(to, l.to.terminal, false)}</p>
        </div>
      </div>
      <p class="leg-meta">
        <img class="icon" src="assets/luggage.png" alt=""> Baggage : Cabin - ${esc(l.cabin)} | Check-in - ${esc(l.checkin)} |
        <img class="icon" src="assets/seat.png" alt=""> Class: ${esc(l.cabinClass)} | ${esc(l.fareType)}
      </p>
    </div>
  </div>`;
}

function buildTravellerTable(journey, coupons) {
  const rows = [];
  state.passengers.forEach((p, pi) => {
    journey.legs.forEach(l => {
      const ticketNumber = `${journey.pnr}/${pi + 1}/${coupons.get(l)}`;
      /* the title keeps its case, the names are printed in caps */
      const name = `${p.title} ${p.first} ${p.last}`.trim()
        .replace(/^(\S+) (.*)$/, (m, title, rest) => `${title} ${rest.toUpperCase()}`);
      rows.push(`<tr>
            <td class="strong">${esc(name)}</td>
            <td>${esc(journey.pnr)}</td>
            <td>${esc(ticketNumber)}</td>
            <td>${esc(p.insurance).replace(' ', '<br>')}</td>
            <td>${esc(p.seat)}</td>
            <td>${esc(p.meal)}</td>
            <td>${esc(p.baggage)}</td>
            <td>${esc(p.status)}</td>
            <td class="col-barcode">${barcodeImg(ticketNumber)}</td>
          </tr>`);
    });
  });

  return `<section class="travellers">
  <h3 class="section-title">Traveller Details</h3>
  <div class="table-scroll">
    <table class="grid travellers-table">
      <thead>
        <tr>
          <th scope="col">Passenger</th>
          <th scope="col">Airline<br>PNR</th>
          <th scope="col">Ticket<br>Number</th>
          <th scope="col">Insurance</th>
          <th scope="col">Seat</th>
          <th scope="col">Meal</th>
          <th scope="col">Baggage</th>
          <th scope="col">Status</th>
          <th scope="col" class="col-barcode"><span class="sr-only">Barcode</span></th>
        </tr>
      </thead>
      <tbody>
        ${rows.join('\n        ')}
      </tbody>
    </table>
  </div>
</section>`;
}

function buildFeeRows() {
  return state.fare.fees
    .filter(f => f.label || f.amount)
    .map(f => `<tr><td>${esc(f.label)}</td><td class="num">&#8377; ${amount(f.amount)}</td></tr>`)
    .join('\n          ');
}

function buildCancelTables() {
  const groups = [];
  state.cancellation.forEach(row => {
    const route = row.auto ? routeLabel(row) : row.route;
    if (!route && !row.window && !row.amount) return;
    const last = groups[groups.length - 1];
    if (last && last.route === route) last.rows.push(row);
    else groups.push({ route, rows: [row] });
  });

  return groups.map(group => {
    const body = group.rows.map((row, i) => `<tr>
              ${i === 0 && group.rows.length > 1 ? `<td rowspan="${group.rows.length}" class="rowspan">${esc(row.type)}</td>` : ''}
              ${i === 0 && group.rows.length === 1 ? `<td class="rowspan">${esc(row.type)}</td>` : ''}
              <td>${esc(row.window)}</td>
              <td class="num">${row.amount ? '&#8377; ' + amount(row.amount) : ''}</td>
            </tr>`).join('\n            ');

    return `<table class="grid cancel-table${group.rows.length === 1 ? ' cancel-table--compact' : ''}">
            <colgroup>
              <col class="col-type"><col class="col-time"><col class="col-air">
            </colgroup>
            <thead>
              <tr>
                <th scope="colgroup" colspan="2">${esc(group.route)}</th>
                <th scope="col">Cancellation Charges</th>
              </tr>
              <tr>
                <th scope="col">Type</th>
                <th scope="col">Cancellation Time</th>
                <th scope="col">Airline</th>
              </tr>
            </thead>
            <tbody>
            ${body}
            </tbody>
          </table>`;
  }).join('\n          ');
}

function buildAirlineContacts() {
  const used = [];
  state.journeys.forEach(j => j.legs.forEach(l => {
    if (!used.includes(l.airlineId)) used.push(l.airlineId);
  }));

  const withContact = used
    .map(id => airlines.find(a => a.id === id))
    .filter(a => a && a.contact);

  const lines = withContact.map((a, i) => {
    const last = i === withContact.length - 1;
    return `<p class="${i === 0 ? 'gap-lg' : 'gap'}">${esc(a.name)} Contact Information: ${esc(a.contact)}${last ? '<br>Please reference the Airline PNR Number when communicating with the airline regarding this booking.' : ''}</p>`;
  });

  if (!lines.length) {
    lines.push('<p class="gap">Please reference the Airline PNR Number when communicating with the airline regarding this booking.</p>');
  }
  return lines.join('\n    ');
}

// ==========================================
// FORMATTING HELPERS
// ==========================================

function airportByCode(code) {
  return airports.find(a => a.iata === code) || { iata: code || '', city: '', name: '' };
}

function legRouteLabel(l) {
  const from = airportByCode(l.from.code);
  const to = airportByCode(l.to.code);
  if (!from.city && !to.city) return '';
  return `${from.city || l.from.code} to ${to.city || l.to.code}`;
}

function routeLabel(cancelRow) {
  const journey = state.journeys.find(j => j.id === cancelRow.journeyId);
  if (!journey || !journey.legs.length) return cancelRow.route || '';
  const from = airportByCode(journey.legs[0].from.code);
  const to = airportByCode(journey.legs[journey.legs.length - 1].to.code);
  return `${from.city || journey.legs[0].from.code}-${to.city || journey.legs[journey.legs.length - 1].to.code}`;
}

function airportLine(airport, terminal, isDeparture) {
  if (!terminal) return esc(airport.name);
  return isDeparture
    ? `${esc(airport.name)} ,<br>Terminal ${esc(terminal)}`
    : `${esc(airport.name)} , Terminal${esc(terminal)}`;
}

function parseDate(value) {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(value || '');
  if (!m) return null;
  return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
}

function parseDateTime(date, time) {
  const d = parseDate(date);
  const t = /^(\d{1,2}):(\d{2})/.exec(time || '');
  if (!d || !t) return null;
  d.setHours(Number(t[1]), Number(t[2]), 0, 0);
  return d;
}

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
const MONTHS_LONG = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December'];

function longDate(value) {
  const d = parseDate(value);
  if (!d) return '';
  return `${WEEKDAYS[d.getDay()]}, ${MONTHS_LONG[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`;
}

function shortDate(value) {
  const d = parseDate(value);
  if (!d) return '';
  return `${WEEKDAYS[d.getDay()]} ${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()}`;
}

function arrivalDate(value) {
  const d = parseDate(value);
  if (!d) return '';
  return `${WEEKDAYS[d.getDay()]}-${d.getDate()}${MONTHS[d.getMonth()]}${d.getFullYear()}`;
}

function formatBookedOn(value) {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{1,2}):(\d{2})/.exec(value || '');
  if (!m) return value || '';
  const d = new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]));
  const hours = Number(m[4]);
  const ampm = hours >= 12 ? 'PM' : 'AM';
  const hour12 = hours % 12 === 0 ? 12 : hours % 12;
  return `${WEEKDAYS[d.getDay()]} ${d.getDate()} ${MONTHS[d.getMonth()]} ${d.getFullYear()} ${hour12}:${m[5]} ${ampm}`;
}

function durationText(l) {
  const from = parseDateTime(l.depart.date, l.depart.time);
  const to = parseDateTime(l.arrive.date, l.arrive.time);
  if (!from || !to) return '';
  const minutes = Math.round((to - from) / 60000);
  if (minutes <= 0) return '';
  const hours = Math.floor(minutes / 60);
  return `${String(hours).padStart(2, '0')}h ${String(minutes % 60).padStart(2, '0')}m`;
}

function fareTotal() {
  const fees = state.fare.fees.reduce((sum, f) => sum + (Number(f.amount) || 0), 0);
  return (Number(state.fare.basic) || 0) + (Number(state.fare.taxes) || 0) + fees - (Number(state.fare.discount) || 0);
}

function amount(value) {
  return String(Math.round(Number(value) || 0));
}

function barcodeImg(text) {
  try {
    const canvas = document.createElement('canvas');
    bwipjs.toCanvas(canvas, {
      bcid: 'code128',
      text: text,
      scale: 2,
      textxalign: 'center',
      backgroundcolor: '#FFFFFF'
    });
    return `<img class="barcode" src="${canvas.toDataURL('image/png')}" alt="Barcode ${escAttr(text)}">`;
  } catch (e) {
    console.error('Barcode generation failed', e);
    return `<div class="barcode" style="font-size:9px;">${esc(text)}</div>`;
  }
}

// ==========================================
// DOM / STATE PLUMBING
// ==========================================

function getPath(object, path) {
  return path.split('.').reduce((node, key) => (node == null ? undefined : node[key]), object);
}

function setPath(object, path, value) {
  const keys = path.split('.');
  const last = keys.pop();
  const target = keys.reduce((node, key) => node[key], object);
  target[last] = value;
}

/* DOMPurify strips the markup; the manual escape is the fallback when the
   CDN copy of DOMPurify is unavailable. */
function esc(value) {
  const text = String(value == null ? '' : value);
  if (window.DOMPurify) return DOMPurify.sanitize(text, { ALLOWED_TAGS: [] });
  return text.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function escAttr(value) {
  return esc(value).replace(/"/g, '&quot;');
}

function renderToIframe(html) {
  const iframe = document.getElementById('preview');
  const doc = iframe.contentWindow.document;
  doc.open();
  doc.write(html);
  doc.close();
  /* Grow the frame to the full document so the preview and the print output
     agree on what is on the sheet. */
  iframe.style.height = Math.max(doc.body.scrollHeight, 1123) + 'px';
}

function validateForm() {
  const form = document.getElementById('ticket-form');
  if (!form.checkValidity()) {
    form.reportValidity();
    return false;
  }
  return true;
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

function disableControls(disabled) {
  document.querySelectorAll('.controls-panel button').forEach(b => {
    if (b.id !== 'btn-print') b.disabled = disabled;
  });
}

function confirmReset() {
  if (confirm('Reset the whole ticket? All entered details will be lost.')) {
    seedDefaults();
    renderAll();
    const iframe = document.getElementById('preview');
    const doc = iframe.contentWindow.document;
    doc.open();
    doc.write('');
    doc.close();
    document.getElementById('btn-print').disabled = true;
    showAlert('Form reset.', 'warning');
  }
}

function printTicket() {
  const iframe = document.getElementById('preview');
  iframe.contentWindow.focus();
  iframe.contentWindow.print();
}
