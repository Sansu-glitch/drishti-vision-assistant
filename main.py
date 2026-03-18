import time
from modules.speech import listen, speak
from modules.vision import detect_objects
from modules.ocr import read_text_from_camera
from modules.scene import describe_scene
from modules.currency import detect_currency

print("Main started...")
speak("Hello! I am your vision assistant. How can I help you?", lang='en')

READ_KEYWORDS = ["read", "text", "what does", "written", "says", "medicine", "prescription", "letter", "word", "पढ़", "पढो"]
DETECT_KEYWORDS = ["detect", "object", "front", "see", "what is", "around", "surroundings", "who", "person", "what are", "सामने", "क्या है"]
SCENE_KEYWORDS = ["describe", "scene", "look", "tell me", "explain", "what is happening", "doing", "what can you see", "बताओ", "देखो"]
CURRENCY_KEYWORDS = ["currency", "note", "money", "rupee", "cash", "how much", "denomination", "नोट", "पैसे", "रुपये"]
EXIT_KEYWORDS = ["exit", "bye", "goodbye", "stop", "quit", "बंद", "रुको"]

# Response messages in different languages
RESPONSES = {
    'en': {
        'read': 'Let me read that for you.',
        'detect': 'Let me check what is around you.',
        'scene': 'Let me look around for you.',
        'currency': 'Let me check the currency note.',
        'exit': 'Goodbye! Stay safe.',
        'unknown': 'Sorry, I did not understand. Please try again.',
        'greeting': 'Hello! I am your vision assistant. How can I help you?'
    },
    'hi': {
        'read': 'मैं आपके लिए पढ़ता हूं।',
        'detect': 'मैं देखता हूं आपके सामने क्या है।',
        'scene': 'मैं आपके आसपास देखता हूं।',
        'currency': 'मैं नोट पहचानता हूं।',
        'exit': 'अलविदा! ध्यान रखें।',
        'unknown': 'माफ करें, मैं समझ नहीं पाया। दोबारा कहें।',
        'greeting': 'नमस्ते! मैं आपका विज़न असिस्टेंट हूं।'
    }
}

def get_response(action, lang):
    if lang in RESPONSES:
        return RESPONSES[lang].get(action, RESPONSES['en'][action])
    return RESPONSES['en'].get(action, RESPONSES['en']['unknown'])

def understand_command(command):
    command = command.lower().strip()
    for word in EXIT_KEYWORDS:
        if word in command:
            return "exit"
    for word in CURRENCY_KEYWORDS:
        if word in command:
            return "currency"
    for word in READ_KEYWORDS:
        if word in command:
            return "read"
    for word in SCENE_KEYWORDS:
        if word in command:
            return "scene"
    for word in DETECT_KEYWORDS:
        if word in command:
            return "detect"
    return "unknown"

while True:
    print("Waiting for command...")
    result = listen(timeout=5)

    if not result:
        continue

    command, lang = result

    if not command or len(command.strip().split()) < 2:
        print("Too short, ignoring:", command)
        continue

    print("You said:", command)
    print("Language detected:", lang)
    action = understand_command(command)
    print("Action:", action)

    response_lang = lang if lang in RESPONSES else 'en'

    if action == "exit":
        speak(get_response('exit', response_lang), lang=response_lang)
        break

    elif action == "read":
        speak(get_response('read', response_lang), lang=response_lang)
        text = read_text_from_camera()
        print("Detected Text:", text)
        speak(text, lang='en')

    elif action == "detect":
        speak(get_response('detect', response_lang), lang=response_lang)
        result = detect_objects()
        print("Detected:", result)
        speak(result, lang='en')

    elif action == "scene":
        speak(get_response('scene', response_lang), lang=response_lang)
        result = describe_scene()
        print("Scene:", result)
        speak(result, lang='en')

    elif action == "currency":
        speak(get_response('currency', response_lang), lang=response_lang)
        result = detect_currency()
        print("Currency:", result)
        speak(result, lang='en')

    else:
        speak(get_response('unknown', response_lang), lang=response_lang)

    time.sleep(1)

print("System stopped safely.")

