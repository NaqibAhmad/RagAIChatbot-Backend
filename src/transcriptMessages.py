from typing import List
from .customTypes import Utterance

def convert_transcript_to_openai_messages(transcript: List[Utterance]):
    messages = []
    for utterance in transcript:
        role = "assistant" if utterance.role == "agent" else "user"
        messages.append({"role": role, "content": utterance.content})
        # print("Messages in convert transcript:",messages)
        
    return messages
    