# WhatsAppFriend
Short experiments, aimed at learning how to finetune Bielik-4.5B-v3.0-Instruct to imitate a fren.

# Project aim
After a few years in game development, I am trying to break into the AI dev space. I have decided that in order to do this, the best way would be to have at least two basic projects to present: [Grounded RAG chatbot](https://loremaster.up.railway.app/), and fine tuning an open weights model to sound more like a person. Due to some slight paranoia over privacy protection, no cloud-based models were used in samples treatment or creation of the synthetic datasets. They did help with scripting, though. That constraint, paired with RTX 2090, is probably the biggest handicap of the project. 

# Simillar attempts
The style cloning was of course done many times, some as one-off projects [1](https://github.com/francescobrigante/Digital-Self-Replica) [2](https://github.com/obattisha/omargpt), some as a continous hobby effort [3](https://github.com/NotYuSheng/Doppelganger). I have not found a simillar one specifically for Polish. 

# Project data
The initial dataset consists of 1,436 messages. All the messages are in Polish. The subjects of the messages are mostly everyday check-ins, there is almost no long-form answers. It is very likely that one thought is split over a few consecutive messages. 

# Detailed prompt approach
The natural benchmark for my LoRA approach would be to try a detailed system prompt, with few shot examples sprinkled in. Here is the example prompt. It is to be noted that the system prompt for all generations is much more compact. This approach appears in the end tests as detailed_system_prompt_model
```# Rola

Pisz po polsku w stylu osoby o profilu:

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

[TUTAJ WSTAW 12 FRAGMENTÓW ROZMÓW] `
```

Loss functions: while computing loss functions of test data is crucial in most LLM post training, it doesn't seem to work here, as the answers were too chaotic to predict.

# Naive approach
The first approach was to just dump the conversations into the LoRA trainer, without masking the speaker and user turns. The conversations are sliced into a rolling window on length 4. That model was called full_chat_dump_model and the hyperparameters used are: 
	gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=8e-5,
	LORA_R = 4
	LORA_A = 8
	TARGET_MODULES = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
	]

# Masking the user turns
The second approach masks the user turns, leaving only the last answer to compute the model loss. The LoRA was trained in two variants: q_proj and k_proj only, and all 4 projections. The reason I meddled with projections so much was because I saw some overfitting during the initial experiments. The models were called masked_chat_dump_q_k_v_o_model, and masked_chat_dump_q_k_model. 
# Reverse instructions
As explained in [This paper](https://arxiv.org/abs/2304.08460), the next logical step would be to turn this data into the Q&A format, typical of chatbot interactions. For the instruction generation, a slightly more capable Bielik-Minitron-7B-v3.0-Instruct was used. Bielik-4.5B-v3.0-Instruct was tried, but it refused to give NULL answers when unsure about the contents of the conversations, often repeating the questions provided to it in the prompt. The synthetic generation was run on a rolling windows of 2, 4 and 6 speaker turns, and then the non-null QA pairs were combined. 


#Grading the models:
The test dataset is
Here is how the eval set loss and test set loss looks like for the selected models:

The other metrics I have used are...., and here are the results:

