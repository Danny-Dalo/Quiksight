/**
 * Client-side file parsing for Quiksight
 * Uses SheetJS (xlsx) to parse CSV/Excel files in the browser
 * Sends pre-parsed JSON to server to reduce CPU load
 */

// SheetJS is loaded via CDN in the HTML

const MAX_FILE_SIZE = 30 * 1024 * 1024; // 30MB
const ALLOWED_EXTENSIONS = ['.csv', '.xlsx', '.xls'];

/**
 * Parse a file using SheetJS and return structured data
 * @param {File} file - The file to parse
 * @returns {Promise<{columns: string[], rows: object[]}>}
 */
async function parseFile(file) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (e) => {
            try {
                const data = new Uint8Array(e.target.result);

                // Parse with SheetJS
                const workbook = XLSX.read(data, {
                    type: 'array',
                    cellDates: true,      // Parse dates properly
                    cellNF: false,        // Don't parse number formats
                    cellText: false,      // Don't generate formatted text
                });

                // Get first sheet only (matching server behavior)
                const sheetName = workbook.SheetNames[0];
                if (!sheetName) {
                    reject(new Error('No sheets found in file'));
                    return;
                }

                const sheet = workbook.Sheets[sheetName];

                // Convert to JSON with header row
                const jsonData = XLSX.utils.sheet_to_json(sheet, {
                    header: 1,     // Return array of arrays first to get headers
                    raw: true,     // Get raw values
                    defval: null,  // Default value for empty cells
                });

                if (jsonData.length === 0) {
                    reject(new Error('File appears to be empty'));
                    return;
                }

                // First row is headers
                const headers = jsonData[0].map((h, i) =>
                    h !== null && h !== undefined ? String(h).trim() : `Column_${i + 1}`
                );

                // Convert remaining rows to objects
                const rows = [];
                for (let i = 1; i < jsonData.length; i++) {
                    const row = jsonData[i];
                    // Skip completely empty rows
                    if (!row || row.every(cell => cell === null || cell === undefined || cell === '')) {
                        continue;
                    }

                    const rowObj = {};
                    headers.forEach((header, index) => {
                        let value = row[index];

                        // Convert Date objects to ISO strings for JSON serialization
                        if (value instanceof Date) {
                            value = value.toISOString();
                        }

                        rowObj[header] = value !== undefined ? value : null;
                    });
                    rows.push(rowObj);
                }

                console.log(`✓ Parsed ${rows.length} rows, ${headers.length} columns`);
                resolve({ columns: headers, rows: rows });

            } catch (error) {
                console.error('Parse error:', error);
                reject(new Error(`Failed to parse file: ${error.message}`));
            }
        };

        reader.onerror = () => {
            reject(new Error('Failed to read file'));
        };

        reader.readAsArrayBuffer(file);
    });
}

/**
 * Validate file before parsing
 * @param {File} file 
 * @returns {{valid: boolean, error?: string}}
 */
function validateFile(file) {
    if (!file) {
        return { valid: false, error: 'No file selected' };
    }

    // Check file size
    if (file.size > MAX_FILE_SIZE) {
        return { valid: false, error: 'File too large. Maximum size is 30MB.' };
    }

    // Check extension
    const fileName = file.name.toLowerCase();
    const ext = '.' + fileName.split('.').pop();
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
        return { valid: false, error: `Invalid file type. Allowed: ${ALLOWED_EXTENSIONS.join(', ')}` };
    }

    return { valid: true };
}

/**
 * Upload parsed data to server
 * @param {File} file - Original file (for metadata)
 * @param {{columns: string[], rows: object[]}} parsedData 
 * @returns {Promise<{success: boolean, session_id?: string, error?: string}>}
 */
async function uploadParsedData(file, parsedData) {
    const payload = {
        filename: file.name,
        file_size: file.size,
        columns: parsedData.columns,
        rows: parsedData.rows,
    };

    console.log(`📤 Sending to server: ${parsedData.rows.length} rows`);

    const response = await fetch('/upload', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
    });

    const result = await response.json();

    if (!response.ok) {
        throw new Error(result.detail || 'Upload failed');
    }

    return result;
}

/**
 * Main upload handler - orchestrates parsing and upload
 * @param {File} file 
 * @param {Function} onProgress - Progress callback: (stage, message) => void
 * @returns {Promise<{success: boolean, session_id?: string}>}
 */
async function handleFileUpload(file, onProgress = () => { }) {
    try {
        // Validate
        onProgress('validating', 'Validating file...');
        const validation = validateFile(file);
        if (!validation.valid) {
            throw new Error(validation.error);
        }

        // Parse
        onProgress('parsing', 'Reading and parsing file...');
        const parsedData = await parseFile(file);

        // Upload
        onProgress('uploading', 'Sending data to server...');
        const result = await uploadParsedData(file, parsedData);

        onProgress('complete', 'Done!');
        return result;

    } catch (error) {
        onProgress('error', error.message);
        throw error;
    }
}

// Export for use in other scripts
window.QuiksightUpload = {
    parseFile,
    validateFile,
    uploadParsedData,
    handleFileUpload,
};
