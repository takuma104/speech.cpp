# A Pocket TTS request speaks a long text in chunks, each from the voice, as the official generate_audio() does

## Context

Pocket TTS was trained on single sentences, so the official `generate_audio()` cuts a text before it speaks
(`split_into_best_sentences()`): it prepares the text (characters replaced, a capital first letter, a period at the
end), tokenizes it, cuts it after the tokens of ".!...?" that its tokenizer gives, but for a decimal period, cuts a
sentence of more than 50 tokens after its tokens of ",;:", and joins the sentences into chunks of at most 50 tokens.
Each chunk is prepared again and spoken from the voice's state, with a fresh decoder whose first 5 ms fade in, for at
most its tokens / 3 + 2 s, and the chunks' audio is joined directly. A sentence that cannot be cut stays one chunk, past
50 tokens. The tools speak a text a sentence at a time on a model whose longest request is under 60 s, cutting it at
their own punctuation and joining the sentences with 0.9 s of silence
([the record of the tools' sentences](on-a-model-whose-request-speaks-less-than-a-minute-speech-tts-and-speech-serve-speak-a-text-a-sentence-at-a-time-and-the-worker-cuts-a-text-refused-as-too-long.md)).

## Decision

- **The family cuts a request's text as the official code does**, with the model's own tokenizer, and speaks the
  chunks in turn in one request: each from the voice's keys and values, with a fresh decoder that fades in its first
  5 ms, and the audio of each passed on as it is made, joined to the one before without silence.
- **A request has no length of its own to refuse.** Its text may be as long as the caller likes, and `max_seconds`
  bounds the speech of the whole; the tools therefore see a model that reads a text whole and pass it in one request.
- **The rules and the 50 tokens come from the model file**, which the converter takes from the configuration and
  `default_parameters.py`, so that a model of other rules, such as the Japanese one, cuts as its own configuration says.

The alternatives were turned down:

- The tools' sentences on Pocket TTS. They cut at other marks than the model's tokens, would split what the official
  code keeps in one chunk, and join with 0.9 s of silence where the official audio has none.
- One sequence for the whole text. The model was trained on single sentences, and the official code never speaks one.
- Each chunk prompted with the audio of the chunk before, which the official code names as an idea and does not do.
  Nothing would check it.

## Consequences

The chunks of a text are the official ones only while the C++ text rules and tokenizer give exactly what the official
code gives; `pocket-tts-text-check` compares them on texts that reach every rule.
