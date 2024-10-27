# Discord-KoboldCPP-Bot
Simple Python script to connect a local model running on KoboldCPP's API to a discord bot.

## Steps:
* Go to the Discord developer portal and create an application. Name it anything you want.
* Under "Bot", create a bot token and save it into a file. You will pass it into the Python script. I will refer to it in the guide token.in
*  Enable the following in the Bot settings:
    - Presence Intent
    - Server Members Intent
    - Message Content Intent
* Go to OAuth2
    - Select "bot" under OAuth2 URL Generator
    - Under bot permissions select "Send Mesages", "View Channels", and "Read Message History"
    - Copy the generate URL and add the bot to your server
* Download the Python script in this directory and save it in a directory.
* The directory should also contain the file with your saved bot token.
* Run your LLM of choice using KoboldCPP. You may have to modify the prompt format in the Python script at the top in the global configuration settings. 
* Make sure you have Python installed as well as any of the libraries in the script using ```pip install```
* Go to the termial and use the command ```python discordbot_LLM.py < token.in``` where token.in is your saved Bot token.
* The bot should be working and will respond to the commands ```!ask``` and ```!chat```.
    - ```!ask``` is a simple input-output command that does not take in any prior messages for context.
    - ```!chat``` is a command that allows you to "chat" with the LLM and will take in prior messages for context.
