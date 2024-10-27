import discord
import requests
import asyncio
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging
from enum import Enum

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Supported prompt formats
class PromptFormat(str, Enum):
    CHATML = "chatml"
    VICUNA = "vicuna"
    ALPACA = "alpaca"
    COMMAND_R = "command-r"
    PHI3 = "phi3"
    MISTRAL = "mistral"
    LLAMA3 = "llama3"
    GEMMA2 = "gemma2"


# Global Configuration
BOT_CONFIG = {
    "MAX_TOKENS": 2048,
    "MAX_MESSAGES": 10,
    "COOLDOWN_SECONDS": 2,
    "CLEANUP_MINUTES": 30,
    "LLM_ENDPOINT": "http://localhost:5001/api/v1/generate",
    "CHAT_MAX_LENGTH": 800,
    "ASK_MAX_LENGTH": 1000,
    "CHAT_TEMPERATURE": 0.75,
    "ASK_TEMPERATURE": 0.5,
    "PROMPT_FORMAT": PromptFormat.CHATML,  # Default format
    "REQUEST_TIMEOUT": 120,  # Increased timeout in seconds
}

@dataclass
class Message:
    content: str
    author_name: str
    timestamp: datetime
    is_bot: bool

class PromptFormatter:
    @staticmethod
    def format_chatml(messages: List[Message], system_prompt: str) -> str:
        prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
        for msg in messages:
            role = "assistant" if msg.is_bot else "user"
            clean_content = msg.content
            if msg.is_bot and clean_content.startswith(f"{msg.author_name}: "):
                clean_content = clean_content[len(f"{msg.author_name}: "):]
            prompt += f"<|im_start|>{role}\n{msg.author_name}: {clean_content}<|im_end|>\n"
        prompt += "<|im_start|>assistant\n"
        return prompt

    @staticmethod
    def format_vicuna(messages: List[Message], system_prompt: str) -> str:
        prompt = f"{system_prompt}\n\n"
        for msg in messages:
            role = "ASSISTANT" if msg.is_bot else "USER"
            clean_content = msg.content
            if msg.is_bot and clean_content.startswith(f"{msg.author_name}: "):
                clean_content = clean_content[len(f"{msg.author_name}: "):]
            prompt += f"{role}: {msg.author_name}: {clean_content}</s>\n"
        prompt += "ASSISTANT:"
        return prompt

    @staticmethod
    def format_alpaca(messages: List[Message], system_prompt: str) -> str:
        prompt = f"{system_prompt}\n\n"
        for msg in messages:
            clean_content = msg.content
            if msg.is_bot and clean_content.startswith(f"{msg.author_name}: "):
                clean_content = clean_content[len(f"{msg.author_name}: "):]
            if msg.is_bot:
                prompt += f"### Response:\n{msg.author_name}: {clean_content}</s>\n\n"
            else:
                prompt += f"### Instruction:\n{msg.author_name}: {clean_content}\n\n"
        prompt += "### Response:\n"
        return prompt

    @staticmethod
    def format_command_r(messages: List[Message], system_prompt: str) -> str:
        prompt = f"<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{system_prompt}<|END_OF_TURN_TOKEN|>"
        for msg in messages:
            token = "<|CHATBOT_TOKEN|>" if msg.is_bot else "<|USER_TOKEN|>"
            clean_content = msg.content
            if msg.is_bot and clean_content.startswith(f"{msg.author_name}: "):
                clean_content = clean_content[len(f"{msg.author_name}: "):]
            prompt += f"<|START_OF_TURN_TOKEN|>{token}{msg.author_name}: {clean_content}<|END_OF_TURN_TOKEN|>"
        prompt += "<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>"
        return prompt

    @staticmethod
    def format_phi3(messages: List[Message], system_prompt: str) -> str:
        prompt = f"<|system|>\n{system_prompt}<|end|>\n"
        for msg in messages:
            role = "assistant" if msg.is_bot else "user"
            clean_content = msg.content
            if msg.is_bot and clean_content.startswith(f"{msg.author_name}: "):
                clean_content = clean_content[len(f"{msg.author_name}: "):]
            prompt += f"<|{role}|>\n{msg.author_name}: {clean_content}<|end|>\n"
        prompt += "<|assistant|>\n"
        return prompt

    @staticmethod
    def format_mistral(messages: List[Message], system_prompt: str) -> str:
        prompt = f"{system_prompt}"
        for msg in messages:
            clean_content = msg.content
            if msg.is_bot and clean_content.startswith(f"{msg.author_name}: "):
                clean_content = clean_content[len(f"{msg.author_name}: "):]
            if msg.is_bot:
                prompt += f"[/INST]{msg.author_name}: {clean_content}</s>"
            else:
                prompt += f"[INST]{msg.author_name}: {clean_content}"
        prompt += "[/INST]"
        return prompt

    @staticmethod
    def format_llama3(messages: List[Message], system_prompt: str) -> str:
        prompt = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"
        for msg in messages:
            role = "assistant" if msg.is_bot else "user"
            clean_content = msg.content
            if msg.is_bot and clean_content.startswith(f"{msg.author_name}: "):
                clean_content = clean_content[len(f"{msg.author_name}: "):]
            prompt += f"<|start_header_id|>{role}<|end_header_id|>\n\n{msg.author_name}: {clean_content}<|eot_id|>"
        prompt += "<|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    @staticmethod
    def format_gemma2(messages: List[Message], system_prompt: str) -> str:
        prompt = f"<start_of_turn>system\n{system_prompt}<end_of_turn>\n"
        for msg in messages:
            role = "model" if msg.is_bot else "user"
            clean_content = msg.content
            if msg.is_bot and clean_content.startswith(f"{msg.author_name}: "):
                clean_content = clean_content[len(f"{msg.author_name}: "):]
            prompt += f"<start_of_turn>{role}\n{msg.author_name}: {clean_content}<end_of_turn>\n"
        prompt += "<start_of_turn>model\n"
        return prompt

class ConversationManager:
    def __init__(self):
        self.max_tokens = BOT_CONFIG["MAX_TOKENS"]
        self.max_messages = BOT_CONFIG["MAX_MESSAGES"]
        self.conversations: Dict[int, List[Message]] = {}
        
    def estimate_tokens(self, text: str) -> int:
        """Rough token estimation"""
        # Count words
        words = len(text.split())
        # Count special characters and punctuation
        special_chars = sum(not c.isalnum() and not c.isspace() for c in text)
        # Account for tokenization of special characters and word pieces
        return int(words * 1.5 + special_chars)
    
    def cleanup_old_conversations(self):
        """Remove conversations older than threshold"""
        current_time = datetime.now()
        threshold = timedelta(minutes=BOT_CONFIG["CLEANUP_MINUTES"])
        for channel_id in list(self.conversations.keys()):
            if self.conversations[channel_id]:
                last_message_time = self.conversations[channel_id][-1].timestamp
                if current_time - last_message_time > threshold:
                    del self.conversations[channel_id]

    def add_message(self, channel_id: int, message: Message):
        """Add message to conversation history"""
        if channel_id not in self.conversations:
            self.conversations[channel_id] = []
        
        self.conversations[channel_id].append(message)
        
        # Trim conversation if needed
        while len(self.conversations[channel_id]) > self.max_messages:
            self.conversations[channel_id].pop(0)

    def get_conversation_context(self, channel_id: int) -> List[Message]:
        """Get relevant conversation context within token limits"""
        if channel_id not in self.conversations:
            return []
        
        context = []
        total_tokens = 0
        
        for msg in reversed(self.conversations[channel_id]):
            msg_tokens = self.estimate_tokens(msg.content)
            if total_tokens + msg_tokens > self.max_tokens:
                break
            context.insert(0, msg)
            total_tokens += msg_tokens
            
        return context

class LLMClient:
    def __init__(self):
        self.endpoint = BOT_CONFIG["LLM_ENDPOINT"]
        self.session = requests.Session()
        self.prompt_format = BOT_CONFIG["PROMPT_FORMAT"]
        
    def build_system_prompt(self, is_chat: bool = True) -> str:
        if is_chat:
            return """You are an AI in a Discord server conversation.
                     - Keep responses concise and engaging
                     - Use appropriate Discord markdown for formatting when needed
                     - Stay within 2000 character limit per message
                     - Maintain context awareness
                     - talk in a casual tone
                     - Format code blocks with appropriate language tags"""
        else:
            return """You are an AI assistant answering a direct question.
                     - Provide clear, focused answers
                     - Use Discord markdown for formatting when appropriate
                     - Stay within 2000 character limit
                     - Format code blocks with appropriate language tags
                     - Be concise but thorough"""

    def format_conversation(self, messages: List[Message]) -> str:
        system_prompt = self.build_system_prompt(True)
        formatter_map = {
            PromptFormat.CHATML: PromptFormatter.format_chatml,
            PromptFormat.VICUNA: PromptFormatter.format_vicuna,
            PromptFormat.ALPACA: PromptFormatter.format_alpaca,
            PromptFormat.COMMAND_R: PromptFormatter.format_command_r,
            PromptFormat.PHI3: PromptFormatter.format_phi3,
            PromptFormat.MISTRAL: PromptFormatter.format_mistral,
            PromptFormat.LLAMA3: PromptFormatter.format_llama3,
            PromptFormat.GEMMA2: PromptFormatter.format_gemma2
        }
    
        if self.prompt_format not in formatter_map:
            logger.warning(f"Unsupported prompt format {self.prompt_format}, falling back to CHATML")
            return PromptFormatter.format_chatml(messages, system_prompt)
            
        return formatter_map[self.prompt_format](messages, system_prompt)

    def format_simple_prompt(self, prompt: str) -> str:
        system_prompt = self.build_system_prompt(False)
        messages = [Message(prompt, "user", datetime.now(), False)]
        return self.format_conversation(messages)

    def clean_response(self, response: str) -> str:
        """Clean up response based on prompt format"""
        cleanup_patterns = {
            PromptFormat.CHATML: "<|im_end|>",
            PromptFormat.VICUNA: "</s>",
            PromptFormat.ALPACA: "</s>",
            PromptFormat.COMMAND_R: "<|END_OF_TURN_TOKEN|>",
            PromptFormat.PHI3: "<|end|>",
            PromptFormat.MISTRAL: "</s>",
            PromptFormat.LLAMA3: "<|eot_id|>",
            PromptFormat.GEMMA2: "<end_of_turn>"
        }
        
        pattern = cleanup_patterns.get(self.prompt_format)
        if pattern and response.endswith(pattern):
            response = response[:-len(pattern)]
            
        return response.strip()

    async def get_response(self, prompt: str, is_chat: bool = True) -> Optional[str]:
        request_body = {
            "max_context_length": BOT_CONFIG["MAX_TOKENS"],
            "max_length": BOT_CONFIG["CHAT_MAX_LENGTH"] if is_chat else BOT_CONFIG["ASK_MAX_LENGTH"],
            "prompt": prompt,
            "temperature": BOT_CONFIG["CHAT_TEMPERATURE"] if is_chat else BOT_CONFIG["ASK_TEMPERATURE"],
            "top_p": 0.9,
            "top_k": 40,
            "rep_pen": 1.15,
            "rep_pen_range": 1024,
            "typical": 1
        }
        
        try:
            response = self.session.post(
                self.endpoint, 
                json=request_body, 
                timeout=BOT_CONFIG["REQUEST_TIMEOUT"]
            )
            response.raise_for_status()
            
            result = response.json()
            if not result.get("results"):
                logger.error("Empty results in LLM response")
                return "I received an invalid response. Please try again."
                
            generated_text = result["results"][0].get("text", "").strip()
            if not generated_text:
                logger.error("Empty text in LLM response")
                return "I received an empty response. Please try again."
            
            # Clean up response based on format
            generated_text = self.clean_response(generated_text)
                
            # Ensure response length is within Discord limits
            if len(generated_text) > 1900:
                # Find the last complete sentence before the cutoff
                last_period = generated_text[:1900].rfind('.')
                if last_period > 0:
                    generated_text = generated_text[:last_period + 1] + "..."
                else:
                    generated_text = generated_text[:1900] + "..."
                
            return generated_text
            
        except requests.exceptions.Timeout:
            logger.error("LLM request timed out")
            return "I apologize, but I'm experiencing some delays. Please try again in a moment."
        except requests.exceptions.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            return "I encountered an error processing your request. Please try again later."
        except (KeyError, IndexError, json.JSONDecodeError) as e:
            logger.error(f"Error parsing LLM response: {e}")
            return "I received an invalid response. Please try again."

class EnhancedChatBot(discord.Client):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conversation_manager = ConversationManager()
        self.llm_client = LLMClient()
        self.command_cooldowns = {}
        
    async def handle_ask_command(self, message: discord.Message):
        """Simple ask command handler"""
        # Check cooldown
        if not self.check_cooldown(message.author.id):
            await message.channel.send("Please wait a few seconds between commands.")
            return
            
        # Extract prompt
        prompt = message.content[5:].strip()  # Remove !ask prefix
        if not prompt:
            await message.channel.send("Please provide a question after `!ask`.")
            return
            
        try:
            async with message.channel.typing():
                # Format prompt and get response
                formatted_prompt = self.llm_client.format_simple_prompt(prompt)
                response_text = await self.llm_client.get_response(formatted_prompt, is_chat=False)
                
                if response_text:
                    await message.channel.send(response_text)
                    
        except discord.HTTPException as e:
            logger.error(f"Discord API error: {e}")
            await message.channel.send("I encountered an error sending the message. Please try again.")
        except Exception as e:
            logger.error(f"Unexpected error in ask command: {e}")
            await message.channel.send("An unexpected error occurred. Please try again later.")

    async def handle_chat_command(self, message: discord.Message):
        """Enhanced chat command handler"""
        # Check cooldown
        if not self.check_cooldown(message.author.id):
            await message.channel.send("Please wait a few seconds between commands.")
            return
            
        # Extract prompt
        prompt = message.content[6:].strip()
        if not prompt:
            await message.channel.send("Please provide a message after `!chat`.")
            return
            
        try:
            async with message.channel.typing():
                # Store current message
                current_msg = Message(
                    content=prompt,
                    author_name=message.author.display_name,
                    timestamp=datetime.now(),
                    is_bot=False
                )
                self.conversation_manager.add_message(message.channel.id, current_msg)
                
                # Get conversation context
                context = self.conversation_manager.get_conversation_context(message.channel.id)
                
                # Get LLM response
                formatted_prompt = self.llm_client.format_conversation(context)
                response_text = await self.llm_client.get_response(formatted_prompt, is_chat=True)
                
                if response_text:
                    # Store bot response in conversation history
                    bot_msg = Message(
                        content=response_text,
                        author_name=self.user.display_name,
                        timestamp=datetime.now(),
                        is_bot=True
                    )
                    self.conversation_manager.add_message(message.channel.id, bot_msg)
                    
                    # Send response
                    await message.channel.send(response_text)
                    
                # Cleanup old conversations periodically
                self.conversation_manager.cleanup_old_conversations()
                
        except discord.HTTPException as e:
            logger.error(f"Discord API error: {e}")
            await message.channel.send("I encountered an error sending the message. Please try again.")
        except Exception as e:
            logger.error(f"Unexpected error in chat command: {e}")
            await message.channel.send("An unexpected error occurred. Please try again later.")
    
    def check_cooldown(self, user_id: int) -> bool:
        """Check and update command cooldown"""
        current_time = datetime.now()
        if user_id in self.command_cooldowns:
            last_used = self.command_cooldowns[user_id]
            if current_time - last_used < timedelta(seconds=BOT_CONFIG["COOLDOWN_SECONDS"]):
                return False
        
        self.command_cooldowns[user_id] = current_time
        return True

    async def on_message(self, message: discord.Message):
        """Message event handler"""
        if message.author == self.user:
            return
            
        if message.content.startswith("!ask"):
            await self.handle_ask_command(message)
        elif message.content.startswith("!chat"):
            await self.handle_chat_command(message)

# Usage
def main():
    
    intents = discord.Intents.default()
    intents.message_content = True
    
    bot = EnhancedChatBot(intents=intents)
    
    # pass input file
    TOKEN = input()

    try:
        bot.run(TOKEN)
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")

if __name__ == "__main__":
    main()
