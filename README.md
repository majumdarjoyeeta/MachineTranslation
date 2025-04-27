# MachineTranslation
# 🌍 Machine Translation (English to Hindi) 🗣️
🔗 **GitHub Repository**: [majumdarjoyeeta/MachineTranslation](https://github.com/majumdarjoyeeta/MachineTranslation)

## 📌 Project Overview
This project develops a **language translation model** 📝 using **Seq2Seq architecture** with **LSTM (Long Short Term Memory)** layers to translate **English sentences** into **Hindi**. The model is trained using **parallel sentences** in English and Hindi from a dataset, and it predicts the corresponding translation for an input sentence 🗣️➡️🇮🇳.

---

## ⚙️ Installation
To run this project, you'll need to install the required libraries:

```bash
pip install tensorflow pandas matplotlib numpy scikit-learn
Ensure you have the necessary GloVe embeddings for word vectors, which are required for this project.

📚 Dataset
We use a dataset of English-Hindi sentence pairs (stored in hin.txt file). Each line in the file contains an English sentence and its corresponding Hindi translation, separated by a tab (\t).

🔧 Project Workflow
1. Data Preprocessing 🧹
Load sentences: Each sentence pair (English-Hindi) is loaded and stored.

Input and Output Sentences: input_sentences (English) and output_sentences (Hindi) are split from the dataset.

Add special tokens: <sos> (start-of-sequence) and <eos> (end-of-sequence) are added to output sentences.

Random Check: A few sentences are printed to check the accuracy of the loaded data.

python
Copy
Edit
input_sentence = line.rstrip().split('\t')[0]
output = line.rstrip().split('\t')[1]
output_sentence = output + ' <eos>'
output_sentence_input = '<sos> ' + output
2. Tokenization and Padding 🔡
Tokenize input (English) and output (Hindi) sentences using Keras's Tokenizer.

Sentences are converted into integer sequences for machine learning models.

Sentences are padded to ensure consistent input/output lengths.

python
Copy
Edit
encoder_input_sequences = pad_sequences(input_integer_seq, maxlen=max_input_len)
decoder_input_sequences = pad_sequences(output_input_integer_seq, maxlen=max_out_len, padding='post')
3. Word Embeddings 🌐
GloVe embeddings are used for representing words as dense vectors.

Download and extract pre-trained GloVe embeddings:

bash
Copy
Edit
wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
unzip -q glove.twitter.27B.zip
These embeddings are used to initialize an Embedding Layer for the model.

4. Model Architecture 🏗️
A Seq2Seq model with LSTM layers is built to perform the translation task:

Encoder: Encodes the input English sentence into a fixed-length context vector.

Decoder: Uses this context vector to predict the Hindi translation word by word.

python
Copy
Edit
encoder_inputs = Input(shape=(max_input_len,))
x = embedding_layer(encoder_inputs)
encoder = LSTM(LSTM_NODES, return_state=True)
encoder_outputs, h, c = encoder(x)
encoder_states = [h, c]

decoder_inputs = Input(shape=(max_out_len,))
decoder_embedding = Embedding(num_words_output, LSTM_NODES)
decoder_inputs_x = decoder_embedding(decoder_inputs)
decoder_lstm = LSTM(LSTM_NODES, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs_x, initial_state=encoder_states)
decoder_dense = Dense(num_words_output, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)
5. Training 🚀
The model is trained using the prepared data. RMSProp optimizer and Categorical Crossentropy loss function are used for training.

python
Copy
Edit
history = model.fit([encoder_input_sequences, decoder_input_sequences], decoder_targets_one_hot,
                    batch_size=BATCH_SIZE, epochs=EPOCHS, validation_split=0.1)
The model's training loss and accuracy are plotted for both training and validation sets. 📈

6. Saving the Model 💾
After training, the model is saved using model.save(). The trained model can then be loaded for further use or evaluation.

python
Copy
Edit
model.save('seq2seq_eng-hindi.h5')
7. Making Predictions 🧑‍💻
Once the model is trained, we can use it to translate new English sentences into Hindi. The decoder model is used for generating translations.

python
Copy
Edit
def translate_sentence(input_seq):
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = word2idx_outputs['<sos>']
    eos = word2idx_outputs['<eos>']
    output_sentence = []

    for _ in range(max_out_len):
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        idx = np.argmax(output_tokens[0, 0, :])

        if eos == idx:
            break

        word = ''
        if idx > 0:
            word = idx2word_target[idx]
            output_sentence.append(word)

        target_seq[0, 0] = idx
        states_value = [h, c]

    return ' '.join(output_sentence)
8. Evaluate the Model 📊
Randomly check the model's predictions against actual translations to ensure the quality of translations.

python
Copy
Edit
input_seq = encoder_input_sequences[i:i+1]
translation = translate_sentence(input_seq)
print('Input Language : ', input_sentences[i])
print('Actual translation : ', output_sentences[i])
print('Hindi translation : ', translation)
📊 Results
Model Accuracy & Loss: The model's accuracy and loss are plotted over the training epochs.

Translation Example: Input sentences are tested for translation into Hindi, and their predictions are printed.

🛠️ Improvements & Future Work
Increase Dataset Size 📦: Using more sentence pairs for training could improve the model's generalization.

Hyperparameter Tuning ⚙️: Experiment with different values for LSTM nodes, batch size, and epochs for better performance.

Deploy Model 🚀: Deploy the trained model as a web app using Flask or FastAPI for real-time translations.

Expand to other Languages 🌏: Extend the model to translate between other language pairs.

🤝 Contribution
Feel free to fork 🍴, star ⭐, or open issues!
Pull Requests (PRs) to improve model performance or contribute to the dataset are welcome! 🎉

👩‍💻 Author
Made with ❤️ by Joyeeta Majumdar
🔗 Visit the GitHub Profile

Let's break language barriers and make communication seamless with this translation model! 🌍💬

markdown
Copy
Edit

### Key Features:
- **Emojis** are used to create an engaging and visually appealing README.
- **Clear sectioning**: Each section has a clear, descriptive heading, making it easy for others to understand the project.
- **Hyperlinks** to external repositories for dataset and GitHub profile.



