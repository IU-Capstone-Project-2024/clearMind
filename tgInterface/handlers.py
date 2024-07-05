from aiogram import Router, types
from aiogram.filters import Command
import requests
import os
import uuid

router = Router()

def forward_to_api(text: str, id: str):
    cookies = {'user_id': str(id)}
    conv_id = str(uuid.uuid4())
    response = requests.post(
        os.getenv('url'),
        cookies=cookies,
        json={
            'input': {'human_input': text},
            'config': {'configurable': {'conversation_id': conv_id}}
        }
    )
    if response.status_code == 200:
        return response.json()
    else:
        return {"error": f"Error: {response.status_code}, {response.text}"}

@router.message(Command("start"))
async def send_welcome(message: types.Message):
    await message.answer(f"Welcome, {message.from_user.full_name}! How can I assist you today?")

@router.message()
async def handle_message(message: types.Message):
    if (message != '/start'):
        while True:
            user_message = message.text

            if user_message.lower() == '/stop':
                break 

            api_response = forward_to_api(user_message, message.from_user.id)
            
            await message.answer(api_response['output']['content'])
            
            message = await message.bot.wait_for('message')
