
AI Medical Diagnosis Bot

Overview

This Python-based AI medical diagnostic bot uses a quantized Llama 2 model to help users identify potential diseases based on their reported symptoms. The bot provides a quick and accessible way to get preliminary medical insights.

Features
    Input symptom analysis using advanced AI
    Disease prediction based on comprehensive medical knowledge
    Quantized Llama 2 model for efficient processing
    Endpoint for symptom-based disease prediction

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Qdrant


