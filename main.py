import time
from modules.nlp import understand_command
from modules.speech import listen, speak
from modules.vision import detect_objects
from modules.ocr import read_text_from_camera
from modules.scene import describe_scene
from modules.currency import detect_currency

print("Main started...")
speak("Hello! I am your vision assistant. How can I help you?", lang='en')

RESPONSES = {
    'en': {
        'read':     'Let me read that for you.',
        'detect':   'Let me check what is around you.',
        'scene':    'Let me look around for you.',
        'currency': 'Let me check the currency note.',
        'exit':     'Goodbye! Stay safe.',
        'unknown':  'Sorry, I did not understand. Please try again.',
        'greeting': 'Hello! I am your vision assistant. How can I help you?'
    },
    'hi': {
        'read':     'मैं आपके लिए पढ़ता हूं।',
        'detect':   'मैं देखता हूं आपके सामने क्या है।',
        'scene':    'मैं आपके आसपास देखता हूं।',
        'currency': 'मैं नोट पहचानता हूं।',
        'exit':     'अलविदा! ध्यान रखें।',
        'unknown':  'माफ करें, मैं समझ नहीं पाया। दोबारा कहें।',
        'greeting': 'नमस्ते! मैं आपका विज़न असिस्टेंट हूं।'
    }
}


def get_response(action, lang):
    if lang in RESPONSES:
        return RESPONSES[lang].get(action, RESPONSES['en'][action])
    return RESPONSES['en'].get(action, RESPONSES['en']['unknown'])


while True:
    print("Waiting for command...")
    listen_result = listen(timeout=5)

    if not listen_result:
        continue

    command, lang = listen_result

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
        ocr_result = read_text_from_camera()
        print("Detected Text:", ocr_result)
        speak(ocr_result, lang='en')

    elif action == "detect":
        speak(get_response('detect', response_lang), lang=response_lang)
        detected_objects = detect_objects()
        print("Detected:", detected_objects)
        speak(detected_objects, lang='en')

    elif action == "scene":
        speak(get_response('scene', response_lang), lang=response_lang)
        scene_description = describe_scene()
        print("Scene:", scene_description)
        speak(scene_description, lang='en')

    elif action == "currency":
        speak(get_response('currency', response_lang), lang=response_lang)
        currency_info = detect_currency()
        print("Currency:", currency_info)
        speak(currency_info, lang='en')

    else:
        speak(get_response('unknown', response_lang), lang=response_lang)

    time.sleep(1)

print("System stopped safely.")