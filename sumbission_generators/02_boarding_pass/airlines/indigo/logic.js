/* =========================================================================
   IndiGo boarding pass — per-airline behaviour for the boarding pass generator.

   The generator has no hook for airline scripts, so template.html pulls this
   file in itself. It runs inside the pass document that main.js assembles with
   document.write(), i.e. after every {{token}} has already been substituted, so
   it only has to do the things token substitution cannot express:

     1. scale the fixed 1630 x 693 ticket down to the A4 preview width,
     2. reformat a handful of values into IndiGo's printed house style,
     3. describe the generated barcode for screen readers.

   Every step is best-effort and independent, so a failure degrades the pass
   rather than blanking it.

   NOTES — deliberate deviations from the source artwork, all forced by what
   the generator can supply:
     * "Your Departure Terminal is T1" and "CPML, CPTR" (Services) are literal
       text. The generator collects no terminal or services field, and a
       terminal cannot be derived from an IATA code alone. Add a "terminal"
       and a "services" input to generator.html, plus matching tokens here,
       when that data becomes available.
     * The stub reuses the main BCBP payload. main.js builds one barcode per
       run, so the artwork's shorter stub payload is not reproduced.
     * The route shows IATA codes ("BLR (T1) TO CCU") rather than city names.
       airports.json holds the names but is not valid JSON (it is two
       concatenated arrays) and nothing in the generator reads it yet.
   ========================================================================= */
(function () {
  "use strict";

  var DESIGN_W = 1630;
  var DESIGN_H = 693;
  var GUTTER = 16;      // page breathing room either side; must match <body> padding in style.css
  var MIN_SCALE = 0.34; // below ~554px wide the ticket is allowed to scroll

  /* ------------------------------------------------------------------ fit
     The ticket is authored at a fixed 1630x693 but the generator previews it in
     a 210mm A4 iframe (~794px), so scale it down and hand the factor to CSS via
     --pass-scale. <body>'s horizontal padding equals GUTTER, so the available
     width derived here is exactly the content width the ticket is centred in. */
  function fit() {
    var frame = document.querySelector(".pass-frame");
    if (!frame) return;

    var available = (document.documentElement.clientWidth || DESIGN_W) - GUTTER * 2;
    var scale = Math.min(1, available / DESIGN_W);
    if (scale < MIN_SCALE) scale = MIN_SCALE;

    frame.style.setProperty("--pass-scale", scale);
    frame.style.width = Math.round(DESIGN_W * scale) + "px";
    frame.style.height = Math.round(DESIGN_H * scale) + "px";
  }

  fit();
  window.addEventListener("resize", fit, { passive: true });
  window.addEventListener("orientationchange", fit);
  if (document.fonts && document.fonts.ready) {
    document.fonts.ready.then(fit);
  }

  /* -------------------------------------------------------------- helpers */
  function each(selector, fn) {
    var nodes = document.querySelectorAll(selector);
    for (var i = 0; i < nodes.length; i++) fn(nodes[i]);
  }

  function textOf(node) {
    return node ? (node.textContent || "").trim() : "";
  }

  function firstOf(selector) {
    return textOf(document.querySelector(selector));
  }

  function titleCase(value) {
    return value.charAt(0) + value.slice(1).toLowerCase();
  }

  function pad2(value) {
    return value.length < 2 ? "0" + value : value;
  }

  /* ---------------------------------------------------------- formatting */

  /* main.js leaves the flight number exactly as typed ("6E442" or "6E 442");
     the airline prints the designator and number as two tokens: "6E 442".
     The lazy prefix requires a letter, so an all-digit input is left alone. */
  function formatFlight(raw) {
    var m = /^([A-Za-z0-9]*?[A-Za-z])(\d+)$/.exec(String(raw).replace(/\s+/g, ""));
    return m ? m[1].toUpperCase() + " " + m[2] : null;
  }

  function formatFlightNumbers() {
    /* main.js substitutes the same raw string into every token, so capture it
       once (already upper-cased by main.js) and reuse it for the <title> that
       template.html also interpolates it into. Matching on the exact raw value
       keeps the two in step without a second pattern to drift out of sync. */
    var raw = null;
    each("[data-flight]", function (node) {
      if (raw === null) raw = textOf(node);
      var flight = formatFlight(raw);
      if (flight) node.textContent = flight;
    });

    if (raw) document.title = document.title.replace(raw, formatFlight(raw) || raw);
  }

  /* main.js renders dates as toLocaleDateString("en-US") and then upper-cases
     them, so {{departDate}} arrives as "MAR 19, 2026" instead of "19 Mar 2026". */
  function formatDates() {
    each("[data-date]", function (node) {
      var m = /^([A-Z]{3})\s+(\d{1,2}),?\s+(\d{4})$/.exec(textOf(node));
      if (m) node.textContent = m[2] + " " + titleCase(m[1]) + " " + m[3];
    });
  }

  /* {{boardTime}} is the raw <input type="time"> value; the pass prints
     "0730 Hrs". Departure time keeps its colon, as in the artwork. */
  function formatBoardingTime() {
    each("[data-board-time]", function (node) {
      var m = /^(\d{1,2}):(\d{2})$/.exec(textOf(node));
      if (m) node.textContent = pad2(m[1]) + m[2] + " Hrs";
    });
  }

  /* main.js hard-codes alt="Barcode" on the image it renders. Re-state the
     same information the artwork described, reading it back off the pass. */
  function describeBarcode() {
    var description = [
      "IATA BCBP boarding pass barcode",
      firstOf("[data-passenger]"),
      "flight " + firstOf("[data-flight]"),
      firstOf("[data-date]"),
      firstOf("[data-from]") + " " + firstOf("[data-depart-time]") + " to " + firstOf("[data-to]"),
      "seat " + firstOf("[data-seat]"),
      "sequence " + firstOf("[data-seq]"),
      "PNR " + firstOf("[data-pnr]")
    ].filter(Boolean).join(", ");

    each(".pass__qr img, .stub__qr img", function (img) {
      img.setAttribute("alt", description);
    });
  }

  /* Run in dependency order: describeBarcode() reads the values the formatters
     above have just rewritten. */
  formatFlightNumbers();
  formatDates();
  formatBoardingTime();
  describeBarcode();
})();
