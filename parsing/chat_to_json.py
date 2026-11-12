"""
Convert chat exports (.txt) into a chat-style dataset for LoRA
fine-tuning with Unsloth.
 
Output format (pretty-printed JSON array, one object per training example):
    [{"conversations": [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
        ...
    ]}, ...]
 
This is the standard "conversations" format Unsloth works with directly:
    from datasets import load_dataset
    from unsloth.chat_templates import get_chat_template
    dataset = load_dataset("json", data_files="training_data.json", split="train")
    tokenizer = get_chat_template(tokenizer, chat_template="chatml")  # or your model's template
    def fmt(batch):
        return {"text": [tokenizer.apply_chat_template(c, tokenize=False)
                         for c in batch["conversations"]]}
    dataset = dataset.map(fmt, batched=True)
"""
 
import json
import random
import re
from datetime import datetime, timedelta
 
# "Name — 05/05/2025 19:18", with the message text on the following lines.
_HEADER_RE = re.compile(r"^(?P<sender>.+?) — (?P<timestamp>\d{2}/\d{2}/\d{4} \d{2}:\d{2})\s*$")
_TIMESTAMP_FORMAT = "%d/%m/%Y %H:%M"

SYSTEM_PROMPT_TRAINING = "Nazywasz się {name} i jesteś {characteristics}. Kontynuuj poniższą konwersację ze swoim przyjacielem w języku polskim, zachowując swój charakterystyczny styl wypowiedzi. "
SYSTEM_PROMPT_CHAT = "Nazywasz się {name} i jesteś {characteristics}. Rozmawiasz ze swoim przyjacielem w języku polskim, zachowując swój charakterystyczny styl wypowiedzi. "
EXTENDED_PROMPT_CHAT = """# Pisz po polsku w stylu osoby o profilu:

- Imię: {name}
- Wiek: {age} lat
- Zajęcie: {occupation}
- Miejsce zamieszkania: {place}
- Dodatkowe informacje: {additional_information}

Rozmawiasz ze swoim bliskim przyjacielem. Znacie się od lat. Wasza komunikacja jest swobodna, bezpośrednia i pozbawiona formalności.

# Cel

Odtwarzaj indywidualny styl komunikacji {name}, nie styl typowego asystenta AI. Najważniejszym wzorcem są autentyczne fragmenty rozmów zamieszczone poniżej.

Naśladuj przede wszystkim:

- **Cadence:** rytm wypowiedzi, tempo rozwijania myśli, dzielenie jednej myśli na kilka krótkich wiadomości.
- **Długość:** typową liczbę słów i proporcje krótkich oraz długich odpowiedzi.
- **Język:** słownictwo, składnię, potocyzmy, skróty, charakterystyczne powiedzenia i błędy.
- **Interpunkcję:** wielkie litery, przecinki, kropki, emoji i ich brak.
- **Osobowość komunikacyjną:** humor, ironię, emocjonalność, bezpośredniość i reakcje na rozmówcę.

Zachowuj naturalną częstotliwość tych cech. Nie stosuj charakterystycznych zwrotów ani wulgaryzmów częściej niż w przykładach.

# Zasady

Odpowiadaj naturalnie i adekwatnie do bieżącego kontekstu. Nie rozwijaj wypowiedzi niepotrzebnie, nie zadawaj sztucznych pytań i nie poprawiaj automatycznie języka.

Jeśli naturalną reakcją jest jedno słowo, napisz jedno słowo. Jeśli myśl powinna zostać podzielona na kilka wiadomości, oddziel je nowymi liniami.

Nie wymyślaj wspólnych wspomnień ani faktów, których nie znasz. Przykłady służą do nauki stylu, a nie kopiowania treści lub wydarzeń.

Nie wyjaśniaj swoich decyzji i nie komentuj sposobu generowania odpowiedzi.

# Przykłady

Poniższe fragmenty pokazują autentyczne rozmowy. **OSOBA** to wzorzec stylu {name}; **PRZYJACIEL** dostarcza kontekstu.

Zachowano kolejność wiadomości. Kolejne wypowiedzi OSOBY mogą stanowić jedną myśl rozbitą na kilka wiadomości.

Naśladuj sposób odpowiadania OSOBY, ale twórz nowe wypowiedzi dopasowane do aktualnej rozmowy.

# Format

Zwracaj wyłącznie wiadomości {name}, bez imienia, nagłówków i komentarzy. Każda osobna wiadomość powinna znajdować się w nowej linii.

# Fragmenty rozmów

{examples}
"""
# Body lines that carry no text worth training on (attachment placeholders).
_SKIP_LINES = {"Image"}


def _parse_file(file_name):
    """Return a list of (timestamp, sender, text) tuples from one export file."""
    messages = []
    with open(file_name, encoding="utf-8-sig") as f:
        for raw_line in f:
            line = raw_line.rstrip("\n").replace("\u200e", "").replace("\u202f", " ")
            match = _HEADER_RE.match(line)
            if match:
                ts = datetime.strptime(match["timestamp"], _TIMESTAMP_FORMAT)
                messages.append((ts, match["sender"].strip(), []))
            elif messages and line.strip() and line.strip() not in _SKIP_LINES:
                messages[-1][2].append(line)

    # A header left with no text (e.g. image only) is not a message to train on.
    return [(ts, sender, "\n".join(lines).strip()) for ts, sender, lines in messages if lines]
 
 
def _split_sessions(messages, gap):
    """Split messages into separate chats wherever the pause exceeds `gap`."""
    sessions, current, last_ts = [], [], None
    for msg in messages:
        if last_ts is not None and msg[0] - last_ts > gap:
            sessions.append(current)
            current = []
        current.append(msg)
        last_ts = msg[0]
    if current:
        sessions.append(current)
    return sessions
 
 
def _rolling_windows(turns, max_turns):
    """
    Cut an alternating user/assistant list into overlapping windows of at most
    `max_turns` messages, each ending with an assistant reply (the target).
    The window slides by one user/assistant exchange:
        [u1, a1, u2, a2], [u2, a2, u3, a3], [u3, a3, u4, a4], ...
    """
    if len(turns) <= max_turns:
        return [turns]
    return [turns[i:i + max_turns] for i in range(0, len(turns) - max_turns + 1, 2)]


def convert_chat_to_training_data(file_names, speaker_names, start_conversation_prompts, speaker_real_name,
                                  speaker_characteristics, output_path="training_data.json", gap_hours=2,
                                  max_turns=4, seed=2137):
    """
    Build a Unsloth-ready chat dataset from chat exports.

    file_names                 -- list of chat .txt export paths
    speaker_names              -- names whose messages become the assistant (bot) replies
    start_conversation_prompts -- generic user prompts; one is picked at random when
                                  a speaker from speaker_names opens the conversation
    speaker_real_name          -- name used in the system prompt
    speaker_characteristics    -- description of the speaker used in the system prompt
    output_path                -- JSON file to write (None = don't write)
    gap_hours                  -- pause longer than this starts a new conversation
    max_turns                  -- max user/assistant messages per example (without the
                                  system prompt); must be even so every window ends
                                  with an assistant reply
    seed                       -- random seed for reproducible prompt choice

    Each session is cut into rolling windows of `max_turns` messages, e.g. for 4:
        [system, u1, a1, u2, a2], [system, u2, a2, u3, a3], ...
    The last assistant reply in each window is what the model learns to predict.

    Returns the list of examples (each a list of {"role", "content"} dicts).
    """
    if max_turns < 2 or max_turns % 2:
        raise ValueError("max_turns must be an even number >= 2")

    rng = random.Random(seed)
    bot_names = {name.strip() for name in speaker_names}
    gap = timedelta(hours=gap_hours)
    system_msg = {"role": "system",
                  "content": SYSTEM_PROMPT_TRAINING.format(name=speaker_real_name,
                                                           characteristics=speaker_characteristics)}
    dataset = []

    for file_name in file_names:
        messages = _parse_file(file_name)
        for session in _split_sessions(messages, gap):
            # Merge consecutive messages from the same side into one turn.
            turns = []
            for _, sender, text in session:
                role = "assistant" if sender in bot_names else "user"
                if turns and turns[-1]["role"] == role:
                    turns[-1]["content"] += "\n" + text
                else:
                    turns.append({"role": role, "content": text})
 
            if not turns:
                continue
            # Bot started the conversation -> answer to a random generic prompt.
            if turns[0]["role"] == "assistant":
                turns.insert(0, {"role": "user",
                                 "content": rng.choice(start_conversation_prompts)})
            # A trailing user message has no answer -> nothing to learn from it.
            if turns[-1]["role"] == "user":
                turns.pop()
            if len(turns) < 2:
                continue

            for window in _rolling_windows(turns, max_turns):
                dataset.append([system_msg] + window)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([{"conversations": conv} for conv in dataset], f,
                      ensure_ascii=False, indent=2)
 
    return dataset
 
 
if __name__ == "__main__":
    data = convert_chat_to_training_data(
        file_names=["./data/asia_chat_1.txt", "./data/asia_chat_2.txt", "./data/asia_chat_3.txt"],
        speaker_names=["anierhej"],
        start_conversation_prompts=["Hej, co tam?", "Siema!", "Co słychać?"],
        speaker_real_name="Ania",
        speaker_characteristics="35 letnią programistką, rozmawiającą ze swoją koleżanką",
        output_path="./data/final_chat_ania.json"
    )
    print(f"{len(data)} conversations written")