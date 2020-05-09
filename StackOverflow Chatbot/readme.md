<h2>StackOverflow assistant bot</h2>

Here is a dialogue chat bot, which will be able to:

* answer programming-related questions (using StackOverflow dataset);
* chit-chat and simulate dialogue on all non programming-related questions.

For a chit-chat mode I have used a pre-trained neural network engine available from [ChatterBot](https://github.com/gunthercox/ChatterBot)

To detect *intent* of users questions we will need two text collections:
- `tagged_posts.tsv` — StackOverflow posts, tagged with one programming language (*positive samples*).
- `dialogues.tsv` — dialogue phrases from movie subtitles (*negative samples*).

For those questions, that have programming-related intent, it will predict the programming language and rank candidates within the tag using embeddings.

For the ranking part, `word_embeddings.tsv` is used. It is trained with StarSpace embeddings, in supervised mode for duplicates detection on the same corpus that is used in search.

The chatbot itself is based on tkinter GUI template taken from https://github.com/aiwithpython/chatbot
