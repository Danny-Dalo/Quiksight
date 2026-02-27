
const urlParams = new URLSearchParams(window.location.search)
const sessionID = urlParams.get("sid");

const chatMessages = document.getElementById("chat-messages");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const textarea = document.getElementById('message-input')


// Removing this cancels the "Enter = Send" functionality
textarea.addEventListener('input', () => {
  textarea.style.height = 'auto'; // Reset height
  textarea.style.height = textarea.scrollHeight + 'px'; // Adjust to content
});



// ===================================================================
(() => {


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
    sendBtn.disabled = !messageInput.value.trim();
    sendBtn.classList.toggle(
      "bg-[#32a3a3]",
      messageInput.value.trim()
    );
    sendBtn.classList.toggle(
      "bg-gray-300",
      !messageInput.value.trim()
    );
    sendBtn.classList.toggle(
      "cursor-not-allowed",
      !messageInput.value.trim()
    );

  });

  // Runs the resize function and button state once the page renders
  function init() {
    resize();
    sendBtn.disabled = !messageInput.value.trim();
  }
  requestAnimationFrame(init);
})();

// ===================================================================



// function to append messages to chat section
function appendMessage(sender, content) {
  const wrapper = document.createElement("div");

  if (sender === "user") {
    wrapper.className = "user-msg";
    wrapper.innerHTML = `<div>${content}</div>`;
  } else {
    wrapper.className = "ai-msg";
    wrapper.innerHTML = content;

    // Style and wrap AI-generated tables
    wrapper.querySelectorAll("table").forEach(table => {
      // Apply Tailwind to the table itself
      table.classList.add("w-full", "border-collapse", "text-sm", "overflow-hidden");
      table.removeAttribute("border");

      // Style thead
      table.querySelectorAll("thead").forEach(thead => {
        thead.classList.add("bg-gray-50");
      });
      table.querySelectorAll("thead th").forEach(th => {
        th.classList.add("px-4", "py-2.5", "text-left", "font-semibold", "text-xs", "uppercase", "tracking-wider", "text-gray-900", "border-b-2", "border-gray-200");
      });

      // Style tbody rows
      table.querySelectorAll("tbody tr").forEach((tr, i) => {
        tr.classList.add("border-b", "border-gray-100", "transition-colors", "hover:bg-gray-100");
        if (i % 2 === 1) tr.classList.add("bg-gray-50/50");
      });
      table.querySelectorAll("tbody td").forEach(td => {
        td.classList.add("px-4", "py-2", "text-gray-700", "align-middle");
      });

      // Wrap in scrollable container with a toolbar
      if (!table.closest(".table-wrap")) {
        const wrap = document.createElement("div");
        wrap.className = "table-wrap rounded-lg border border-gray-200 shadow-sm my-3 overflow-hidden";

        // Toolbar row above the table
        const toolbar = document.createElement("div");
        toolbar.className = "flex justify-end px-2 py-1.5 bg-gray-50 border-b border-gray-200";

        const dlBtn = document.createElement("button");
        dlBtn.className = "inline-flex items-center gap-1 px-2 py-1 rounded-md text-xs text-gray-500 cursor-pointer transition-all hover:bg-gray-200 hover:text-gray-900";
        dlBtn.title = "Download as CSV";
        dlBtn.innerHTML = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="w-3.5 h-3.5">
  <path fill-rule="evenodd" d="M12 2.25a.75.75 0 0 1 .75.75v11.69l3.22-3.22a.75.75 0 1 1 1.06 1.06l-4.5 4.5a.75.75 0 0 1-1.06 0l-4.5-4.5a.75.75 0 1 1 1.06-1.06l3.22 3.22V3a.75.75 0 0 1 .75-.75Zm-9 13.5a.75.75 0 0 1 .75.75v2.25a1.5 1.5 0 0 0 1.5 1.5h13.5a1.5 1.5 0 0 0 1.5-1.5V16.5a.75.75 0 0 1 1.5 0v2.25a3 3 0 0 1-3 3H5.25a3 3 0 0 1-3-3V16.5a.75.75 0 0 1 .75-.75Z" clip-rule="evenodd" />
</svg>CSV`;
        dlBtn.addEventListener("click", () => downloadTableCSV(table));

        toolbar.appendChild(dlBtn);
        wrap.appendChild(toolbar);

        // Scrollable table area
        const scrollArea = document.createElement("div");
        scrollArea.className = "overflow-x-auto";

        table.parentNode.insertBefore(wrap, table);
        scrollArea.appendChild(table);
        wrap.appendChild(scrollArea);
      }
    });
  }

  chatMessages.appendChild(wrapper);
  chatMessages.scrollTop = chatMessages.scrollHeight;
}


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



async function sendMessage() {
  const msg = messageInput.value.trim();
  if (!msg) return;         // Get the user's message, if there's no message, then nothing happens

  appendMessage("user", msg);
  messageInput.value = "";    // User's message is tagged for user styling

  appendMessage("ai", '<span class="loading loading-dots loading-md"></span>');      // AI thinking animation

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

    const data = await res.json();    // Saves the AI response to a variable called 'data'
    chatMessages.lastChild.remove(); // Remove "Thinking..."

    // Check if the response indicates an error (e.g., 404, 500, etc.)
    if (res.status === 404) {
      const errorMessage = "Conversation Not Found. Session may be expired or deleted.";
      appendMessage("ai", `<span style="color:red;">${errorMessage}</span>`);
      return;
    }

    // AI response is the text returned
    const aiResponse = data.response.text || "No response received.";
    appendMessage("ai", aiResponse);

    // CATCH AND THROW ANY ERRORS THAT MAY COME UP IN A USER-FRIENDLY WAY
  } catch (err) {
    chatMessages.lastChild.remove();
    appendMessage("ai", `<span style="color:red;">Oops! Something went wrong. Please try sending your message again.</span>`);
  }
}

sendBtn.addEventListener("click", sendMessage);   //when the send button is clicked, trigger the function to send message to AI
messageInput.addEventListener("keypress", e => {  // 'Enter' key sends the message as well 
  if (e.key === "Enter") sendMessage();
});


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
        appendMessage("ai", msg.content);
      }
    });

  } catch (err) {
    console.log("Could not load chat history:", err);
  }
}

// Load history when page loads
loadChatHistory();
// ===================== LOAD CHAT HISTORY ON PAGE LOAD =====================