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
  });

  // Runs the resize function once the page renders
  requestAnimationFrame(resize);
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
    }

    chatMessages.appendChild(wrapper);
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
  // function to append messages to chat section


  
  async function sendMessage() {                         
    const msg = messageInput.value.trim();    
    if (!msg) return;         // Get the user's message, if there's no message, then nothing happens

    appendMessage("user", msg);
    messageInput.value = "";    // User's message is tagged for user styling

    appendMessage("ai", '<span class="loading loading-dots loading-md"></span>');      // AI thinking animation

    // API Call
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

      let aiResponse = "";             // By default, the AI response is nothing
      if (data.response.text) {aiResponse += `
          ${data.response.text}`;   // aiResponse becomes the text reponse that the AI returns
      }
      // if (data.response.text) {aiResponse += `
      //   <div style="padding: 6px; margin-bottom: 8px;">
      //     ${data.response.text}
      //   </div>`;   // aiResponse becomes the text reponse that the AI returns
      // }

      
      // // MONITORING THE CODE AI GENERATES
      // if (data.response.code) {
      //       aiResponse += `
      //   <pre style="border: 1px solid blue; padding: 6px; margin-bottom: 8px; background:#f9f9f9;">
      //     <code>${data.response.code}</code>
      //   </pre>`;    // if there is code generated, it's added to the AI's response
      // }

      // // MONITORING THE EXECUTION RESULT PROVIDED
      if (data.response.execution_results) {
        aiResponse += `
              ${data.response.execution_results}`;   // if the code has  results, it's added to the AI's response as well
      }

      // if (data.response.execution_results) {
      //   aiResponse += `
      //     <div style="border: 1px solid green; padding: 6px; margin-bottom: 8px; white-space: pre-wrap; font-family: monospace;">
      //         ${data.response.execution_results}
      //     </div>`;   // if the code has  results, it's added to the AI's response as well
      // }
      

      // AI response is added to the ai tag so the div is populated with the aiResponse
     
      appendMessage("ai", aiResponse);    

      // CATCH AND THROW ANY ERRORS THAT MAY COME UP IN A USER-FRIENDLY WAY
    } catch (err) {               
      chatMessages.lastChild.remove();       
      appendMessage("ai", `<span style="color:red;">Oops! Something went wrong. Please try sending your message again.</span>`);
    
      //if an error occurs, it removes the last AI response and replaces it with the error message
    }
  }

  sendBtn.addEventListener("click", sendMessage);   //when the send button is clicked, trigger the function to send message to AI
  messageInput.addEventListener("keypress", e => {  // 'Enter' key sends the message as well 
    if (e.key === "Enter") sendMessage();
  });




















