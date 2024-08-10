# Quantized Encoder Decoder Transformer

Quantized encoder-decoder architecture for decoder-only style next token prediction. Adds bidirectionality to instruct style large language models to mitigate decoder-only unidirectional context limits to enhance:

- Deeper understanding of full conversational context
- Ability to ground responses in structured knowledge
- Reduce the likelihood of generating outputs that are not grounded in the input data ie. hallucinations
