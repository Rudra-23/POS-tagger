The project uses hidden Markov model to generate part of speech tags for sentences.

Implementations:
1. Vocabulary generation.
2. Creating transistion and emission states.
3. Greedy decoding.
4. Viterbi decoding.
5. Adding tags.

Validations on dev data:

Threshold of unknown word: 2. Size of Vocabulary: 23183, Occurrences of <unk> : 20011. No of parameters in emissions: 30303, no of parameters in transitions: 1392.

Greedy: total: 131768, correct: 123207, accuracy: 93.50%
Viterbi: total: 131768, correct: 123357, accuracy: 93.62%