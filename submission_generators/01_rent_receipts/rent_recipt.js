// ==========================================
// UTILITY FUNCTIONS
// ==========================================

/**
 * Displays a toast notification message
 * @param {string} message - The text to display
 * @param {string} type - 'error' or 'success'
 */
function showToast(message, type = 'success') {
    const toast = document.getElementById("toast");
    toast.textContent = message;
    toast.className = type === 'error' ? 'error show' : 'success show';
    setTimeout(() => { toast.className = toast.className.replace("show", ""); }, 3000);
}

/**
 * Generates a unique Receipt ID
 * Format: RR-YYYYMMDD-XXXX
 */
function generateReceiptId() {
    const receiptDate = document.getElementById('receiptDate').value;
    console.log(receiptDate)
    const now = new Date(receiptDate);
    const dateStr = now.toISOString().slice(0,10).replace(/-/g, ''); // YYYYMMDD
    const random = Math.floor(1000 + Math.random() * 9000); // 4 digit random
    return `RR-${dateStr}-${random}`;
}

/**
 * Validates Indian PAN format (Basic regex check)
 * 5 letters, 4 digits, 1 letter
 */
function isValidPan(pan) {
    const panRegex = /[A-Z]{5}[0-9]{4}[A-Z]{1}/;
    return panRegex.test(pan);
}

// ==========================================
// NUMBER TO WORDS (INDIAN SYSTEM)
// ==========================================

/**
 * Converts a number to Indian English words (Lakhs/Crores)
 * Supports up to Crores and decimal Paise.
 * @param {number|string} amount
 * @returns {string} - Amount in words
 */
function numberToIndianRupees(amount) {
    if (isNaN(amount) || amount === 0) return "Zero Only";

    // Split integer and decimal part
    const amountStr = parseFloat(amount).toFixed(2);
    const [intPart, decPart] = amountStr.split('.');

    const units = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine"];
    const teens = ["Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"];
    const tens = ["", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"];

    function convertHundreds(n) {
        if (n === 0) return "";
        if (n < 10) return units[n];
        if (n < 20) return teens[n - 10];
        if (n < 100) return tens[Math.floor(n / 10)] + (n % 10 !== 0 ? " " + units[n % 10] : "");
        return units[Math.floor(n / 100)] + " Hundred" + (n % 100 !== 0 ? " " + convertHundreds(n % 100) : "");
    }

    // Indian Numbering System grouping:
    // ...XX,YY,ZZ,WWW.VV
    // Crores, Lakhs, Thousands, Hundreds
    let num = parseInt(intPart, 10);
    let words = "";

    if (num >= 10000000) { // Crores
        const crores = Math.floor(num / 10000000);
        words += convertHundreds(crores) + " Crore ";
        num %= 10000000;
    }
    if (num >= 100000) { // Lakhs
        const lakhs = Math.floor(num / 100000);
        words += convertHundreds(lakhs) + " Lakh ";
        num %= 100000;
    }
    if (num >= 1000) { // Thousands
        const thousands = Math.floor(num / 1000);
        words += convertHundreds(thousands) + " Thousand ";
        num %= 1000;
    }
    if (num > 0) { // Hundreds
        words += convertHundreds(num);
    }

    let result = "Rupees " + words.trim();

    // Handle Paise
    if (parseInt(decPart, 10) > 0) {
        const paiseNum = parseInt(decPart, 10);
        let paiseWords = "";
        if (paiseNum < 10) paiseWords = units[paiseNum];
        else if (paiseNum < 20) paiseWords = teens[paiseNum - 10];
        else paiseWords = tens[Math.floor(paiseNum / 10)] + (paiseNum % 10 !== 0 ? " " + units[paiseNum % 10] : "");

        result += " and " + paiseWords.trim() + " Paise";
    }

    return result.trim() + " Only";
}

// ==========================================
// CORE LOGIC
// ==========================================

// Set default dates on load
window.addEventListener('DOMContentLoaded', () => {
    const today = new Date();
    document.getElementById('receiptDate').valueAsDate = today;
    document.getElementById('monthYear').value = today.toISOString().slice(0, 7);
});

function generateReceipt() {
    // 1. Retrieve Values
    const tenantName = document.getElementById('tenantName').value.trim();
    const tenantPan = document.getElementById('tenantPan').value.trim();
    const propertyAddress = document.getElementById('propertyAddress').value.trim();
    const landlordName = document.getElementById('landlordName').value.trim();
    const landlordPan = document.getElementById('landlordPan').value.trim().toUpperCase();
    const amount = parseFloat(document.getElementById('amount').value);
    const receiptDate = document.getElementById('receiptDate').value;
    const monthYear = document.getElementById('monthYear').value;
    const paymentMode = document.getElementById('paymentMode').value;

    // 2. Basic Validation
    if (!tenantName || !tenantPan || !propertyAddress || !landlordName || !landlordPan || !amount || !receiptDate || !monthYear) {
        showToast("Please fill in all required fields.", "error");
        return;
    }

    // 3. Tax Compliance Validation (Section 10(13A))
    // Rule: If annual rent > 1,00,000, Landlord PAN is mandatory.
    const annualRent = amount * 12;
    const isPanMandatory = annualRent > 100000;

    if (isPanMandatory) {
        if (!landlordPan) {
            showToast("Error: Annual rent exceeds ₹1 Lakh. Landlord PAN is mandatory.", "error");
            document.getElementById('landlordPan').focus();
            return;
        }
        if (!isValidPan(landlordPan)) {
            showToast("Error: Invalid PAN format (e.g., ABCDE1234F).", "error");
            document.getElementById('landlordPan').focus();
            return;
        }
    }

    // 4. Update Receipt Preview
    document.getElementById('r-id').textContent = generateReceiptId();
    document.getElementById('r-tenantName').textContent = `${tenantName.toUpperCase()} [PAN: ${tenantPan.toUpperCase()}]`;
    document.getElementById('r-property').textContent = propertyAddress;
    document.getElementById('r-amountFigures').textContent = amount.toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    document.getElementById('r-amountWords').textContent = numberToIndianRupees(amount);
    document.getElementById('r-mode').textContent = paymentMode;

    // Format date for display (DD-MM-YYYY)
    const dateObj = new Date(receiptDate);
    document.getElementById('r-date').textContent = `Date: ${dateObj.toLocaleDateString('en-IN')}`;

    // Handle Month/Year Display
    const [yr, mo] = monthYear.split('-');
    const monthDate = new Date(yr, mo - 1);
    const monthName = monthDate.toLocaleString('en-IN', { month: 'long' });
    document.getElementById('r-period').textContent = `${monthName} ${yr}`;

    // Handle PAN Display
    const panDisplay = document.getElementById('r-pan');
    if (landlordPan) {
        panDisplay.textContent = `${landlordName.toUpperCase()} [PAN: ${landlordPan.toUpperCase()}]`;
        document.getElementById('panRow').style.display = 'flex';
    } else {
        // Hide PAN row entirely if not provided (and not mandatory) to keep receipt clean
        document.getElementById('panRow').style.display = 'none';
    }

    // 5. Handle Signature Image Rotation
    const sigImg = document.getElementById('landlordSignature');
    // Generate random angle between -10 and -1 degrees
    const minAngle = -10;
    const maxAngle = -1;
    // Math.random() is 0 to 1.
    // (max - min + 1) is range size. Add min to shift.
    const randomAngle = Math.floor(Math.random() * (maxAngle - minAngle + 1)) + minAngle;

    // Apply rotation and show the image
    sigImg.style.transform = `rotate(${randomAngle}deg)`;
    sigImg.style.display = 'block';

    // 6. Enable Print Button
    document.getElementById('printBtn').disabled = false;

    showToast("Receipt generated successfully!");

    // Scroll to preview on mobile
    if(window.innerWidth < 900) {
        document.querySelector('.preview-section').scrollIntoView({behavior: 'smooth'});
    }
}

function printReceipt() {
    // Check if receipt is actually generated
    if(document.getElementById('printBtn').disabled) {
        showToast("Please generate receipt first.", "error");
        return;
    }
    window.print();
}

function resetForm() {
    if(confirm("Are you sure you want to reset? This will clear current details.")) {
        document.getElementById('receiptForm').reset();

        // Reset dates to today
        const today = new Date();
        document.getElementById('receiptDate').valueAsDate = today;
        document.getElementById('monthYear').value = today.toISOString().slice(0, 7);

        // Reset Preview
        document.getElementById('r-id').textContent = "RR-PENDING";
        document.getElementById('r-tenantName').textContent = "[Tenant Name]";
        document.getElementById('r-property').textContent = "[Property Address]";
        document.getElementById('r-amountFigures').textContent = "[0.00]";
        document.getElementById('r-amountWords').textContent = "[Rupees in Words]";
        document.getElementById('r-mode').textContent = "[Mode]";
        document.getElementById('r-pan').textContent = "[PAN Number]";
        document.getElementById('r-date').textContent = "[Date]";
        document.getElementById('panRow').style.display = 'flex';

        document.getElementById('printBtn').disabled = true;
        showToast("Form reset.");
    }
}
