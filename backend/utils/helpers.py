import re

def option_letter_to_text(letter, answer_option_text):
    if not letter or not answer_option_text:
        return ""

    options = {}
    for line in answer_option_text.strip().splitlines():
        if "." in line:
            parts = line.strip().split(".", 1)
            if len(parts) == 2:
                key = parts[0].strip().upper()
                val = parts[1].strip()
                options[key] = val

    return options.get(letter.upper(), "")