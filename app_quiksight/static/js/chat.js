
const urlParams = new URLSearchParams(window.location.search)
const sessionID = urlParams.get("sid");

const chatMessages = document.getElementById("chat-messages");
const chatScroller = document.getElementById("chat-container"); // the scrollable .qs-chat-thread
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const textarea = document.getElementById('message-input')


// Removing this cancels the "Enter = Send" functionality
if (textarea) {
  textarea.addEventListener('input', () => {
    textarea.style.height = 'auto'; // Reset height
    textarea.style.height = textarea.scrollHeight + 'px'; // Adjust to content
  });
}



// ===================================================================
(() => {
  if (!messageInput || !sendBtn) return;

  // Function to resize the textarea dynamically
  function resize() {
    messageInput.style.height = 'auto';
    const max = parseFloat(getComputedStyle(messageInput).maxHeight) || 9999;
    const h = Math.min(messageInput.scrollHeight, max);
    messageInput.style.height = h + 'px';
    messageInput.style.overflowY = messageInput.scrollHeight > max ? 'auto' : 'hidden';
  }

  // Whenever the user types into the textarea, it runs the resize function
  messageInput.addEventListener('input', () => {
    resize();
    const hasText = !!messageInput.value.trim();
    sendBtn.disabled = !hasText;
    sendBtn.classList.toggle("enabled", hasText);

  });

// Runs the resize function and button state once the page renders
  function init() {
    resize();
    sendBtn.disabled = !messageInput.value.trim();
  }
  requestAnimationFrame(init);

  // Suggested questions logic
  const suggestBtns = document.querySelectorAll(".suggest-btn");
  if (suggestBtns.length > 0) {
    suggestBtns.forEach(btn => {
      btn.addEventListener("click", () => {
        messageInput.value = btn.innerText;
        messageInput.focus();
        // trigger input event to adapt size and button states
        messageInput.dispatchEvent(new Event('input', { bubbles: true }));
      });
    });
  }

})();

// ===================================================================



// function to append messages to chat section
function appendMessage(sender, content, charts = []) {
  const wrapper = document.createElement("div");

  if (sender === "user") {
    wrapper.className = "user-msg";
    wrapper.innerHTML = `<div>${content}</div>`;
  } else {
    wrapper.className = "ai-msg";
    wrapper.innerHTML = content;

    // Remove any <!-- chart --> comments from the DOM (we pair by index, not position)
    const walker = document.createTreeWalker(wrapper, NodeFilter.SHOW_COMMENT);
    const chartComments = [];
    while (walker.nextNode()) {
      if (walker.currentNode.textContent.trim() === "chart") {
        chartComments.push(walker.currentNode);
      }
    }
    chartComments.forEach(c => c.remove());

    // Style all tables
    wrapper.querySelectorAll("table").forEach(table => {
      table.classList.add("w-full", "border-collapse", "text-sm");
      table.removeAttribute("border");

      table.querySelectorAll("thead").forEach(thead => thead.classList.add("bg-gray-50"));
      table.querySelectorAll("thead th").forEach(th =>
        th.classList.add("px-4", "py-2.5", "text-left", "font-semibold", "text-xs",
          "uppercase", "tracking-wider", "text-gray-900", "border-b-2", "border-gray-200")
      );
      table.querySelectorAll("tbody tr").forEach((tr, i) => {
        tr.classList.add("border-b", "border-gray-100", "transition-colors", "hover:bg-gray-100");
        if (i % 2 === 1) tr.classList.add("bg-gray-50/50");
      });
      table.querySelectorAll("tbody td").forEach(td =>
        td.classList.add("px-4", "py-2", "text-gray-700", "align-middle")
      );

      // Truncate tables with more than 10 rows
      const MAX_VISIBLE = 10;
      const allRows = table.querySelectorAll("tbody tr");
      if (allRows.length > MAX_VISIBLE) {
        // Hide rows beyond the limit
        allRows.forEach((tr, i) => {
          if (i >= MAX_VISIBLE) tr.classList.add("hidden", "qs-overflow-row");
        });

        // Add a footer row showing count + toggle
        const tfoot = document.createElement("tfoot");
        const footRow = document.createElement("tr");
        const footCell = document.createElement("td");
        const colCount = table.querySelectorAll("thead th").length || 1;
        footCell.setAttribute("colspan", colCount);
        footCell.className = "px-4 py-2.5 text-xs text-gray-700 bg-gray-50 border-t border-gray-200";

        const label = document.createElement("span");
        label.textContent = `Showing ${MAX_VISIBLE} of ${allRows.length} rows`;


        footCell.appendChild(label);
        footRow.appendChild(footCell);
        tfoot.appendChild(footRow);
        table.appendChild(tfoot);
      }
    });

    // Pair tables with charts by index: table[0] ↔ charts[0], table[1] ↔ charts[1], etc.
    const allTables = wrapper.querySelectorAll("table");
    let chartIdx = 0;

    allTables.forEach(table => {
      if (table.closest(".qs-tabbed")) return;

      const hasChart = chartIdx < charts.length;

      const container = document.createElement("div");
      container.className = "qs-tabbed rounded-lg border border-gray-200 shadow-sm my-3 overflow-hidden";

      // Toolbar
      const toolbar = document.createElement("div");
      toolbar.className = "flex items-center justify-between px-3 py-1.5 bg-gray-50 border-b border-gray-200";

      const tabs = document.createElement("div");
      tabs.className = "flex gap-1";

      const tableTab = makeTab("Table", true);
      tabs.appendChild(tableTab);

      let chartTab = null;
      if (hasChart) {
        chartTab = makeTab("Chart", false);
        tabs.appendChild(chartTab);
      }
      toolbar.appendChild(tabs);

      // CSV download button
      const dlBtn = document.createElement("button");
      dlBtn.className = "inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs text-gray-500 cursor-pointer transition-all hover:bg-gray-200 hover:text-gray-900";
      dlBtn.title = "Download as CSV";
      dlBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-3.5 h-3.5">
        <path fill-rule="evenodd" d="M12 2.25a.75.75 0 0 1 .75.75v11.69l3.22-3.22a.75.75 0 1 1 1.06 1.06l-4.5 4.5a.75.75 0 0 1-1.06 0l-4.5-4.5a.75.75 0 1 1 1.06-1.06l3.22 3.22V3a.75.75 0 0 1 .75-.75Zm-9 13.5a.75.75 0 0 1 .75.75v2.25a1.5 1.5 0 0 0 1.5 1.5h13.5a1.5 1.5 0 0 0 1.5-1.5V16.5a.75.75 0 0 1 1.5 0v2.25a3 3 0 0 1-3 3H5.25a3 3 0 0 1-3-3V16.5a.75.75 0 0 1 .75-.75Z" clip-rule="evenodd"/>
      </svg>CSV`;
      dlBtn.addEventListener("click", () => downloadTableCSV(table));
      toolbar.appendChild(dlBtn);
      container.appendChild(toolbar);

      // Table panel
      const tablePanel = document.createElement("div");
      tablePanel.className = "qs-panel overflow-x-auto";
      table.parentNode.insertBefore(container, table);
      tablePanel.appendChild(table);
      container.appendChild(tablePanel);

      // Chart panel (paired by index)
      if (hasChart && chartTab) {
        const ci = chartIdx;
        chartIdx++;

        const chartPanel = document.createElement("div");
        chartPanel.className = "qs-panel hidden p-2";

        const plotDiv = document.createElement("div");
        plotDiv.className = "w-full";
        plotDiv.style.minHeight = "360px";
        chartPanel.appendChild(plotDiv);
        container.appendChild(chartPanel);

        let rendered = false;

        tableTab.addEventListener("click", () => {
          setActiveTab(tableTab, chartTab);
          tablePanel.classList.remove("hidden");
          chartPanel.classList.add("hidden");
        });

        chartTab.addEventListener("click", () => {
          setActiveTab(chartTab, tableTab);
          tablePanel.classList.add("hidden");
          chartPanel.classList.remove("hidden");

          if (!rendered) {
            try {
              const figureData = JSON.parse(charts[ci]);
              Plotly.react(plotDiv, figureData.data, figureData.layout, {
                responsive: true,
                displaylogo: false,
                modeBarButtonsToRemove: ["sendDataToCloud"]
              });
              rendered = true;
            } catch (e) {
              plotDiv.innerHTML = `<p class="text-sm text-red-500 p-4">Could not render chart.</p>`;
            }
          }
        });
      }
    });

    // Render any remaining charts that weren't paired with a table (standalone)
    while (chartIdx < charts.length) {
      const plotDiv = document.createElement("div");
      plotDiv.className = "w-full my-3 rounded-lg border border-gray-200 shadow-sm p-2";
      plotDiv.style.minHeight = "360px";
      wrapper.appendChild(plotDiv);

      try {
        const figureData = JSON.parse(charts[chartIdx]);
        Plotly.react(plotDiv, figureData.data, figureData.layout, {
          responsive: true,
          displaylogo: false,
        });
      } catch (e) {
        plotDiv.innerHTML = `<p class="text-sm text-red-500 p-4">Could not render chart.</p>`;
      }
      chartIdx++;
    }
  }

  chatMessages.appendChild(wrapper);
  chatScroller.scrollTop = chatScroller.scrollHeight;
}

function makeTab(label, active) {
  const btn = document.createElement("button");
  btn.textContent = label;
  btn.className = active
    ? "qs-tab px-3 py-1 rounded-md text-xs font-medium bg-white border border-gray-200 text-gray-800 shadow-sm"
    : "qs-tab px-3 py-1 rounded-md text-xs font-medium text-gray-400 hover:text-gray-700 transition-colors";
  return btn;
}

function setActiveTab(active, inactive) {
  active.className = "qs-tab px-3 py-1 rounded-md text-xs font-medium bg-white border border-gray-200 text-gray-800 shadow-sm";
  inactive.className = "qs-tab px-3 py-1 rounded-md text-xs font-medium text-gray-400 hover:text-gray-700 transition-colors";
}
// function to append messages to chat section


// Convert an HTML table element to a CSV string
function tableToCSV(table) {
  const rows = [];

  // Header row
  const headers = [];
  table.querySelectorAll("thead th").forEach(th => {
    headers.push('"' + th.textContent.trim().replace(/"/g, '""') + '"');
  });
  if (headers.length) rows.push(headers.join(","));

  // Data rows
  table.querySelectorAll("tbody tr").forEach(tr => {
    const cells = [];
    tr.querySelectorAll("td").forEach(td => {
      cells.push('"' + td.textContent.trim().replace(/"/g, '""') + '"');
    });
    rows.push(cells.join(","));
  });

  return rows.join("\n");
}


// Trigger a CSV file download from a table element
function downloadTableCSV(table) {
  const csv = tableToCSV(table);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "quiksight_export.csv";
  link.click();
  URL.revokeObjectURL(url);
}
// function to append messages to chat section



// Adding User-friendly error messages by HTTP status code
const ERROR_MESSAGES = {
  404: "This conversation could not be found. Your session may have expired — please try uploading your file again.",
  500: "Something went wrong on our end. Please try sending your message again.",
  502: "We're having trouble connecting to the AI service. Please try again in a moment.",
  504: "The AI is taking too long to respond. Please try sending your message again.",
  413: "Your request was too large to process. Try asking a simpler question.",
  429: "Too many requests — please wait a moment and try again.",
};

// Common technical error patterns
const TECHNICAL_PATTERNS = [
  /ExecutionError:\s/i,
  /ChartError:\s/i,
  /Traceback \(most recent call last\)/i,
  /\bFile ".*", line \d+/i,
  /\bNameError:/i,
  /\bTypeError:/i,
  /\bValueError:/i,
  /\bKeyError:/i,
  /\bIndexError:/i,
  /\bAttributeError:/i,
  /\bSyntaxError:/i,
  /\bModuleNotFoundError:/i,
  /\bImportError:/i,
  /\bRuntimeError:/i,
  /\bZeroDivisionError:/i,
  /\[INTERNAL TOOL ERROR/i,
];

/**
 * Checks if the AI response text contains leaked technical error patterns.
 * Returns a clean, user-friendly  message if it does, otherwise returns the original text.
 */
function sanitizeAIResponse(text) {
  if (!text || typeof text !== "string") return text;

  for (const pattern of TECHNICAL_PATTERNS) {
    if (pattern.test(text)) {
      return "<p>I ran into a hiccup while processing your request. Could you try rephrasing your question or asking again?</p>";
    }
  }
  return text;
}

async function sendMessage() {
  const msg = messageInput.value.trim();
  if (!msg) return;         // Get the user's message, if there's no message, then nothing happens

  appendMessage("user", msg);
  messageInput.value = "";    // User's message is tagged for user styling

  appendMessage("ai", '<div class="qs-dots"><span></span><span></span><span></span></div>');      // AI thinking animation

  // The API call is made to the chat endpoint based on the session id. This call is made whenever a message is sent through the input
  try {
    // Sending the message as a JSON string to the chat endpoint
    const res = await fetch(`/chat?sid=${sessionID}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: msg
      })
    });

    chatMessages.lastChild.remove(); // Remove "Thinking..."

    // Check if the response indicates an error (e.g., 404, 500, etc.)
    if (res.status === 404) {
      const errorMessage = "Conversation Not Found. Session may be expired or deleted.";
      appendMessage("ai", `<span style="color:red;">${errorMessage}</span>`);
      return;
    }

    const data = await res.json();    // Saves the AI response to a variable called 'data'

    // AI response is the text returned, with charts from the backend
    const rawResponse = data.response?.text || "No response received.";
    const aiResponse = sanitizeAIResponse(rawResponse);
    const charts = data.response?.charts || [];
    appendMessage("ai", aiResponse, charts);

    // CATCH AND THROW ANY ERRORS THAT MAY COME UP IN A USER-FRIENDLY WAY
  } catch (err) {
    chatMessages.lastChild.remove();
    appendMessage("ai", `<p class="text-red-500">Oops! Something went wrong. Please try sending your message again.</p>`);
  }
}

if (sendBtn) {
  sendBtn.addEventListener("click", sendMessage);   //when the send button is clicked, trigger the function to send message to AI
}
if (messageInput) {
  messageInput.addEventListener("keypress", e => {  // 'Enter' key sends the message as well 
    if (e.key === "Enter") sendMessage();
  });
}


// ===================== LOAD CHAT HISTORY ON PAGE LOAD =====================
async function loadChatHistory() {
  if (!sessionID) return;

  try {
    const res = await fetch(`/chat/history?sid=${sessionID}`);
    if (!res.ok) return;

    // return if there has been no message sent
    const data = await res.json();
    if (!data.messages || data.messages.length === 0) return;

    // Render each message
    data.messages.forEach(msg => {
      if (msg.role === "user") {
        appendMessage("user", msg.content);
      } else if (msg.role === "ai") {
        appendMessage("ai", msg.content, msg.charts || []);
      }
    });

  } catch (err) {
    console.log("Could not load chat history:", err);
  }
}

// Load history when page loads
loadChatHistory();
// ===================== LOAD CHAT HISTORY ON PAGE LOAD =====================


// ===================== CHAT SESSION SIDEBAR =====================
const sidebar = document.getElementById('sessions-sidebar');
const sidebarToggle = document.getElementById('sidebar-toggle');
const closeSidebar = document.getElementById('close-sidebar');
const sidebarOverlay = document.getElementById('sidebar-overlay');
const sessionsList = document.getElementById('sessions-list');
const sessionsLoading = document.getElementById('sessions-loading');
const sessionsEmpty = document.getElementById('sessions-empty');

let sessionsLoaded = false;

function openSidebar() {
  sidebar.classList.remove('-translate-x-full');
  sidebarOverlay.classList.remove('hidden');
  // Small delay to allow display block to apply before opacity transition
  setTimeout(() => {
    sidebarOverlay.classList.remove('opacity-0');
  }, 10);
  
  if (!sessionsLoaded) {
    loadSessions();
  }
}

function closeSidebarPanel() {
  sidebar.classList.add('-translate-x-full');
  sidebarOverlay.classList.add('opacity-0');
  setTimeout(() => {
    sidebarOverlay.classList.add('hidden');
  }, 300); // match transition duration
}

if (sidebarToggle) sidebarToggle.addEventListener('click', openSidebar);
if (closeSidebar) closeSidebar.addEventListener('click', closeSidebarPanel);
if (sidebarOverlay) sidebarOverlay.addEventListener('click', closeSidebarPanel);

async function loadSessions() {
  if (!sessionsLoading || !sessionsEmpty || !sessionsList) return;
  
  sessionsLoading.classList.remove('hidden');
  sessionsEmpty.classList.add('hidden');
  
  // Remove any previously rendered session cards
  document.querySelectorAll('.session-card').forEach(el => el.remove());
  
  try {
    const res = await fetch('/sessions');
    if (!res.ok) throw new Error('Failed to fetch sessions');
    const data = await res.json();
    
    sessionsLoading.classList.add('hidden');
    
    if (!data.sessions || data.sessions.length === 0) {
      sessionsEmpty.classList.remove('hidden');
      return;
    }
    
    sessionsLoaded = true;
    
    data.sessions.forEach(session => {
      const card = document.createElement('div');
      card.className = `session-card group block w-full text-left p-3 rounded-xl border ${session.session_id === sessionID ? 'bg-cyan-50/50 border-cyan-200 shadow-sm' : 'bg-white border-gray-100 shadow-sm hover:border-gray-300 hover:shadow-md'} transition-all relative cursor-pointer`;
      
      const link = document.createElement('a');
      link.href = `/chat?sid=${session.session_id}`;
      link.className = "absolute inset-0 z-0";
      card.appendChild(link);
      
      const header = document.createElement('div');
      header.className = "flex justify-between items-start mb-1";
      
      const fileIcon = document.createElement('div');
      fileIcon.className = "flex items-center gap-2 font-medium text-gray-800 text-sm truncate pr-6";
      fileIcon.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4 text-cyan-600 flex-shrink-0">
          <path stroke-linecap="round" stroke-linejoin="round" d="M19.5 14.25v-2.625a3.375 3.375 0 00-3.375-3.375h-1.5A1.125 1.125 0 0113.5 7.125v-1.5a3.375 3.375 0 00-3.375-3.375H8.25m0 12.75h7.5m-7.5 3H12M10.5 2.25H5.625c-.621 0-1.125.504-1.125 1.125v17.25c0 .621.504 1.125 1.125 1.125h12.75c.621 0 1.125-.504 1.125-1.125V11.25a9 9 0 00-9-9z" />
        </svg>
        <span class="truncate" title="${session.file_name}">${session.file_name}</span>
      `;
      
      header.appendChild(fileIcon);
      card.appendChild(header);
      
      const meta = document.createElement('div');
      meta.className = "text-xs text-gray-500 mt-1 flex justify-between";
      meta.innerHTML = `
        <span>${session.num_rows} rows • ${session.file_size}</span>
        <span>${session.upload_date}</span>
      `;
      card.appendChild(meta);
      
      // Delete button
      const delBtn = document.createElement('button');
      delBtn.className = "absolute top-2 right-2 p-1.5 text-gray-400 hover:text-red-500 hover:bg-red-50 rounded-md transition-colors z-10 opacity-0 group-hover:opacity-100";
      delBtn.innerHTML = `
        <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" class="w-4 h-4">
          <path stroke-linecap="round" stroke-linejoin="round" d="M14.74 9l-.346 9m-4.788 0L9.26 9m9.968-3.21c.342.052.682.107 1.022.166m-1.022-.165L18.16 19.673a2.25 2.25 0 01-2.244 2.077H8.084a2.25 2.25 0 01-2.244-2.077L4.772 5.79m14.456 0a48.108 48.108 0 00-3.478-.397m-12 .562c.34-.059.68-.114 1.022-.165m0 0a48.11 48.11 0 013.478-.397m7.5 0v-.916c0-1.18-.91-2.164-2.09-2.201a51.964 51.964 0 00-3.32 0c-1.18.037-2.09 1.022-2.09 2.201v.916m7.5 0a48.667 48.667 0 00-7.5 0" />
        </svg>
      `;
      delBtn.title = "Delete session";
      
      delBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        e.preventDefault();
        
        if (!confirm('Are you sure you want to delete this session?')) return;
        
        delBtn.innerHTML = '<span class="qs-spinner !w-3 !h-3 !border-t-red-500 !border-red-200"></span>';
        
        try {
          const res = await fetch(`/sessions/${session.session_id}`, { method: 'DELETE' });
          if (!res.ok) throw new Error('Delete failed');
          
          card.style.transform = "scale(0.95)";
          card.style.opacity = "0";
          setTimeout(() => {
            card.remove();
            if (document.querySelectorAll('.session-card').length === 0) {
              sessionsEmpty.classList.remove('hidden');
            }
            if (session.session_id === sessionID) {
              window.location.href = '/';
            }
          }, 200);
        } catch (err) {
          console.error(err);
          alert('Failed to delete session');
          delBtn.innerHTML = '...'; 
        }
      });
      
      card.appendChild(delBtn);
      sessionsList.appendChild(card);
    });
    
  } catch (err) {
    console.error("Could not load sessions:", err);
    sessionsLoading.classList.add('hidden');
    sessionsEmpty.classList.remove('hidden');
    sessionsEmpty.textContent = "Failed to load sessions.";
  }
}
// ===================== SIDEBAR LOGIC =====================