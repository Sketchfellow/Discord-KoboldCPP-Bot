
# Discord-KoboldCPP-Bot

This Python script connects a local Large Language Model (LLM) running on KoboldCPP's API to a Discord bot, enabling interactive conversations and Q&A within your Discord server.

## Features

* **Interactive Chat:** Use the `!chat` command to engage in a conversation with the LLM. The bot maintains context and remembers previous messages within a defined limit.
* **Simple Q&A:** Use the `!ask` command for direct questions to the LLM without retaining conversation history.  Ideal for quick queries.
* **Customizable:** Configure various parameters like maximum tokens, message history length, cooldown period, and prompt format to suit your needs.
* **KoboldCPP Integration:** Designed to work seamlessly with KoboldCPP's API, allowing you to leverage locally hosted LLMs.
* **Multiple Prompt Formats:** Supports various prompt formats including ChatML, Llama, Vicuna, Mistral, Gemma, and more. 
* **Automatic Cleanup:**  Periodically removes old conversation histories to manage memory usage.
* **Error Handling:**  Includes robust error handling to manage issues like network timeouts, API errors, and invalid responses.

## Prerequisites

* **Python 3.x:** Ensure you have Python 3 installed on your system.
* **Required Libraries:** Install the necessary Python libraries using pip:
  ```bash
  pip install discord
  ```
* **KoboldCPP:** Set up and run KoboldCPP with your desired LLM.
* **Discord Bot Token:** Create a Discord bot and obtain its token (see setup instructions below).

## Setup

1. **Create a Discord Application:**
   - Go to the [Discord Developer Portal](https://discord.com/developers/applications) and create a new application.
2. **Create a Bot:**
   - Navigate to the "Bot" tab within your application and create a new bot. Copy the bot token and save it securely (e.g., in a `token.txt` file).
3. **Enable Privileged Gateway Intents:**
   - In the "Bot" settings, enable the following gateway intents:
     - Presence Intent
     - Server Members Intent
     - Message Content Intent
4. **Add Bot to Your Server:**
   - Go to the "OAuth2" tab.
   - Select "bot" under "OAuth2 URL Generator."
   - Under "Bot Permissions," select:
     - Send Messages
     - View Channels
     - Read Message History
   - Copy the generated URL and open it in your browser to add the bot to your server.
5. **Configure the Script:**
   - Download `discordbot_LLM.py`.
   - Open the script and adjust the `BOT_CONFIG` dictionary at the top to match your KoboldCPP setup and desired settings (e.g., LLM endpoint, prompt format, max tokens).
6. **Run the Bot:**
   - Open your terminal, navigate to the directory containing `discordbot_LLM.py` and your token file, and run:
     ```bash
     python discordbot_LLM.py < token.txt 
     ```

## Usage

Once the bot is running, you can use the following commands in your Discord server:

* **`!chat [your message]`:**  Starts or continues a conversation with the LLM.
* **`!ask [your question]`:** Asks the LLM a direct question without maintaining conversation history.

Note: This bot script is **not** meant for large-scale deployments.

## Example

```
!chat Hello there!
!chat How are you doing today?
!ask What is the capital of France?
```


## Troubleshooting

* **Bot Not Responding:** Verify that KoboldCPP is running correctly and that the API endpoint in the `BOT_CONFIG` is accurate.  Check the console for any error messages.
* **Rate Limiting:** If you're sending messages too frequently, you might encounter rate limits. Adjust the `COOLDOWN_SECONDS` in the `BOT_CONFIG` to increase the delay between commands.
* **Token Issues:** Make sure you are using the correct bot token and that it has the necessary permissions.

